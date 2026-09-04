#!/bin/bash
#------------------------------------------------------------------
# submit_cgyro_nersc.sh
#
# PURPOSE:
#  Submit one NERSC (Perlmutter) Slurm job per input.cgyro produced
#  by generate_data.sh (see ./cgyro_inputs and ./cgyro_inputs_fast).
#
#  The per-job batch file is generated exactly the way
#  gacode_qsub + platform/qsub/qsub.PERLMUTTER_CPU generate it, but
#  every simulation directory is submitted as its own sbatch job
#  inside a loop (no job arrays, no bundling).
#
# LAYOUT EXPECTED:
#  <input-root>/batch-XXX/cgyro/input-YYY/input.cgyro
#
# USAGE:
#  ./job_scripts/submit_cgyro_nersc.sh [options]
#  For example:
#  ./job_scripts/submit_cgyro_nersc.sh -p ./cgyro_inputs -queue shared -s
#
#  Run without -s to print diagnostics + the first batch file only
#  (dry run); add -s to actually submit.
#------------------------------------------------------------------

#=============================================================
# Defaults (mirrors gacode_qsub defaults where applicable)
#
INPUT_ROOT=./cgyro_inputs
CODE=cgyro
QUEUE=regular          # NERSC "regular" priority
WALLTIME=0:15:00       # 15 minute wallclock
CONSTRAINT=cpu
SUB_FLAG=0
SKIP_DONE=0
MAX_JOBS=0             # 0 = no limit
SLEEP_SEC=0            # throttle between sbatch calls
REPO=
nmpi=16
nomp=1
ENV_SCRIPT=

# Memory request (used by the shared QOS; optional elsewhere).
# Precedence is -mem then -mem-per-cpu, as in gacode_qsub.
MEMPERNODE=
MEMPERCPU=

# Perlmutter CPU: 512 GB and 256 cores per node -> 2 GB/core.  Requesting
# 2 GB/core in the shared QOS makes the memory fraction match the core
# fraction, so the job is charged on cores rather than on memory.
MEM_PER_CORE_GB=2
# The shared QOS is limited to half a node.
SHARED_MAX_CORES=128

usage () {
  cat <<'EOF'

Usage:   submit_cgyro_nersc.sh [options]

         -s
         Submit the generated batch scripts.  Otherwise only print
         diagnostics and a sample batch file (dry run).

         -p <path>
         Root directory holding the generated inputs.
         [default: ./cgyro_inputs]

         -code <code>
         Code to run (cgyro).  [default: cgyro]

         -n <MPI tasks>
         Number of MPI tasks per job.  [default: 16]

         -nomp <threads>
         OpenMP threads per MPI task.  [default: 1]

         -w <wallclock>
         Wallclock limit per job.  [default: 0:15:00]

         -queue <name>
         Queue/QOS name: regular, preempt, debug, premium, shared.
         [default: regular]
         With "shared" the batch file omits -N (nodes are not
         allocated exclusively) and adds --mem, so the job is
         charged for the cores it actually uses.

         -mem <n>[unit]
         Memory required per node.  Default unit MB; Slurm accepts
         [K|M|G|T] with no space before the unit.
         [shared QOS default: MEM_PER_CORE_GB * (MPI tasks * threads)]

         -mem-per-cpu <n>[unit]
         Memory required per cpu.  Ignored if -mem is also given.

         -C <constraint>
         Slurm constraint.  [default: cpu]

         -repo <name>
         NERSC repository (account).  Auto-detected via iris.

         -env <file>
         Extra file to source inside each batch job (e.g. your
         GACODE setup script) before launching the code.

         -skip-done
         Skip simulation directories that already contain
         out.cgyro.info (i.e. previously completed runs).

         -max <n>
         Submit at most <n> jobs.  [default: all]

         -sleep <sec>
         Sleep this many seconds between sbatch calls.  [default: 0]

EOF
  exit 1
}

#=============================================================
# Parse command line options
#
while [[ $# -gt 0 ]] ; do
  case "$1" in

  -s) SUB_FLAG=1 ;;

  -p) shift ; INPUT_ROOT=$1 ;;

  -code) shift ; CODE=$1 ;;

  -n) shift ; nmpi=$1 ;;

  -nomp) shift ; nomp=$1 ;;

  -w) shift ; WALLTIME=$1 ;;

  -queue) shift ; QUEUE=$1 ;;

  -C) shift ; CONSTRAINT=$1 ;;

  -repo) shift ; REPO=$1 ;;

  -mem) shift ; MEMPERNODE=$1 ;;

  -mem-per-cpu) shift ; MEMPERCPU=$1 ;;

  -env) shift ; ENV_SCRIPT=$1 ;;

  -skip-done) SKIP_DONE=1 ;;

  -max) shift ; MAX_JOBS=$1 ;;

  -sleep) shift ; SLEEP_SEC=$1 ;;

  -h|-help|--help) usage ;;

  *) echo "ERROR (submit_cgyro_nersc.sh): Incorrect syntax: $1" ; exit 1 ;;

  esac
  shift
done

if [[ ! -d $INPUT_ROOT ]] ; then
   echo "ERROR: (submit_cgyro_nersc.sh) Input root $INPUT_ROOT not found."
   exit 1
fi

# Absolute path to the simulation root (SIMROOT in gacode_qsub terms)
SIMROOT=$(cd "$INPUT_ROOT" && pwd)

#=============================================================
# Repository (account) resolution -- same logic as gacode_qsub
#
if [[ -z $REPO ]] ; then
   if hash iris 2> /dev/null; then
      # NERSC
      x=`iris`
      y=($x)
      # The 10th element should be the first repo in the list
      REPO=${y[10]}
   else
      REPO=null_repo
   fi
fi

#=============================================================
# Task/thread layout -- same logic as gacode_mpi_tool
#
if [[ -n ${GACODE_ROOT:-} && -n ${GACODE_PLATFORM:-} \
      && -f $GACODE_ROOT/platform/build/make.inc.$GACODE_PLATFORM ]] ; then
   export `grep _NODE $GACODE_ROOT/platform/build/make.inc.$GACODE_PLATFORM`
   export "`grep IDENT $GACODE_ROOT/platform/build/make.inc.$GACODE_PLATFORM`"
   . $GACODE_ROOT/shared/bin/gacode_mpi_tool
else
   # Fall back to Perlmutter CPU node values so the script can be
   # inspected off-platform.
   echo "WARNING: GACODE_ROOT/GACODE_PLATFORM unset; assuming Perlmutter CPU."
   IDENTITY="NERSC Perlmutter CPU"
   CORES_PER_NODE=256
   NUMAS_PER_NODE=8

   numa=$NUMAS_PER_NODE
   mpinuma=$((CORES_PER_NODE / NUMAS_PER_NODE))
   mpinuma=$((mpinuma / nomp))

   mpinode=$((mpinuma * numa))
   if [[ $nmpi -lt $mpinode ]]; then
      mpinode=$nmpi
   fi
   if [[ $nmpi -lt $mpinuma ]]; then
      mpinuma=$nmpi
      numa=1
   fi

   nodes=$((nmpi / mpinode))
   if [[ $nmpi -gt $((nodes * mpinode)) ]]; then
      nodes=$((nodes + 1))
      echo "WARNING: Using partial node"
   fi

   cores_requested=$((nodes * CORES_PER_NODE))
   cores_used=$((nomp * nmpi))
fi

#=============================================================
# Resolve double specification of memory (as in gacode_qsub)
#
if [[ -n $MEMPERNODE && -n $MEMPERCPU ]]; then
   echo "WARNING: Both -mem and -mem-per-cpu specified."
   echo "Applying Precedence -mem=$MEMPERNODE."
   MEMPERCPU=
fi

#=============================================================
# shared QOS: half-node maximum, no exclusive node allocation
#
SHARED=0
if [[ $QUEUE == shared || $QUEUE == shared_interactive || $QUEUE == shared_overrun ]] ; then
   SHARED=1

   cores_per_job=$((nmpi * nomp))

   if [[ $cores_per_job -gt $SHARED_MAX_CORES ]] ; then
      echo "ERROR: (submit_cgyro_nersc.sh) The $QUEUE QOS is limited to half a node"
      echo "       ($SHARED_MAX_CORES cores), but -n $nmpi -nomp $nomp needs $cores_per_job."
      echo "       Reduce -n/-nomp or use -queue regular."
      exit 1
   fi

   # Default the memory request to match the core fraction of the node.
   if [[ -z $MEMPERNODE && -z $MEMPERCPU ]] ; then
      MEMPERNODE=$((cores_per_job * MEM_PER_CORE_GB))GB
   fi

   # A shared job never spans nodes, so -N is dropped from the batch file.
   nodes=1
   cores_requested=$cores_per_job
fi

#=============================================================
# Collect every input.cgyro under the input root
#
mapfile -t INPUT_FILES < <(find "$SIMROOT" -type f -name "input.$CODE" | sort)

if [[ ${#INPUT_FILES[@]} -eq 0 ]] ; then
   echo "ERROR: (submit_cgyro_nersc.sh) No input.$CODE found under $SIMROOT"
   exit 1
fi

echo "INFO: (submit_cgyro_nersc.sh) Job layout diagnostics"
echo
echo "       identity: $IDENTITY"
echo " cores per node: $CORES_PER_NODE"
echo " numas per node: $NUMAS_PER_NODE"
echo
echo " simulation root         : $SIMROOT"
echo " input.$CODE files found : ${#INPUT_FILES[@]}"
echo " account (repo)          : $REPO"
echo " queue                   : $QUEUE"
echo " wallclock               : $WALLTIME"
echo " cores requested / job   : $cores_requested"
echo " cores used / job        : $cores_used"
echo " total MPI tasks         : $nmpi"
echo " MPI tasks/node          : $mpinode"
echo " OpenMP threads/MPI task : $nomp"
if [[ $SHARED -eq 1 ]] ; then
   echo " nodes / job             : shared (-N omitted)"
else
   echo " nodes / job             : $nodes"
fi
if [[ -n $MEMPERNODE ]] ; then
   echo " memory per node         : $MEMPERNODE"
elif [[ -n $MEMPERCPU ]] ; then
   echo " memory per cpu          : $MEMPERCPU"
fi
echo

#=============================================================
# Loop: one batch file + one sbatch submission per simulation
#
njob=0
nskip=0

for input_file in "${INPUT_FILES[@]}" ; do

   SIMDIR=$(dirname "$input_file")
   # LOCDIR is the path relative to SIMROOT, e.g. batch-000/cgyro/input-000
   LOCDIR=${SIMDIR#$SIMROOT/}

   if [[ $SKIP_DONE -eq 1 && -f $SIMDIR/out.$CODE.info ]] ; then
      nskip=$((nskip + 1))
      continue
   fi

   if [[ $MAX_JOBS -gt 0 && $njob -ge $MAX_JOBS ]] ; then
      echo "INFO: Reached -max $MAX_JOBS; stopping."
      break
   fi

   # Slurm job names cannot usefully carry '/'
   JOBNAME=${LOCDIR//\//_}

   bfile=$SIMDIR/batch.src

   # ---- Batch file, generated as in platform/qsub/qsub.PERLMUTTER_CPU ----
   echo "#!/bin/bash -l" > $bfile
   echo "#SBATCH -J $JOBNAME" >> $bfile
   echo "#SBATCH -A $REPO" >> $bfile
   echo "#SBATCH -C $CONSTRAINT" >> $bfile
   echo "#SBATCH -o $SIMDIR/batch.out" >> $bfile
   echo "#SBATCH -e $SIMDIR/batch.err" >> $bfile
   echo "#SBATCH -q $QUEUE" >> $bfile
   echo "#SBATCH -t $WALLTIME" >> $bfile
   if [[ $SHARED -eq 0 ]] ; then
      # Exclusive node allocation, exactly as qsub.PERLMUTTER_CPU does.
      echo "#SBATCH -N $nodes" >> $bfile
   fi
   echo "#SBATCH -n $nmpi" >> $bfile
   echo "#SBATCH -c $nomp" >> $bfile
   if [[ -n $MEMPERNODE ]] ; then
      echo "#SBATCH --mem=$MEMPERNODE" >> $bfile
   elif [[ -n $MEMPERCPU ]] ; then
      echo "#SBATCH --mem-per-cpu=$MEMPERCPU" >> $bfile
   fi
   if [[ -n $ENV_SCRIPT ]] ; then
      echo "source $ENV_SCRIPT" >> $bfile
   fi
   echo 'export SLURM_CPU_BIND="cores"' >> $bfile
   echo "$CODE -e $LOCDIR -n $nmpi -nomp $nomp -numa $numa -mpinuma $mpinuma -p $SIMROOT" >> $bfile

   njob=$((njob + 1))

   if [[ $SUB_FLAG -eq 0 ]] ; then
      if [[ $njob -eq 1 ]] ; then
         echo " sample batch file ($bfile):"
         echo "-----------------------------------------------"
         cat $bfile
         echo "-----------------------------------------------"
         echo
      fi
      echo "[dry run] would submit: $LOCDIR"
   else
      if hash sbatch 2> /dev/null ; then
         echo -n "[$njob/${#INPUT_FILES[@]}] $LOCDIR -> "
         sbatch $bfile
      else
         echo "ERROR: (submit_cgyro_nersc.sh) No sbatch command recognized."
         exit 1
      fi
      if [[ $SLEEP_SEC -gt 0 ]] ; then
         sleep $SLEEP_SEC
      fi
   fi

done

echo
if [[ $SUB_FLAG -eq 0 ]] ; then
   echo "INFO: Dry run complete. $njob job(s) would be submitted (skipped: $nskip)."
   echo "      Re-run with -s to submit."
else
   echo "INFO: Submitted $njob job(s) (skipped: $nskip)."
fi
