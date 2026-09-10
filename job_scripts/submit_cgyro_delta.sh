#!/bin/bash
#------------------------------------------------------------------
# submit_cgyro_delta.sh
#
# PURPOSE:
#  Submit one NCSA Delta Slurm job per input.cgyro produced by
#  generate_data.sh (see ./cgyro_inputs and ./cgyro_inputs_fast).
#
#  Delta analogue of submit_cgyro_nersc.sh.  The per-job batch file is
#  generated the way gacode_qsub + platform/qsub/qsub.DELTA_CPU generate
#  it, but every simulation directory is submitted as its own sbatch job
#  inside a loop (no job arrays, no bundling).
#
# DELTA SPECIFICS:
#  - CPU nodes: 2 x AMD EPYC 7763, 128 cores, 256 GB (2 GB/core), 8 NUMAs
#  - partitions: cpu (48 h max), cpu-interactive (1 h, 4 nodes),
#    cpu-preempt (48 h, half charge) -- "-p", not "-q"
#  - node-sharing is the DEFAULT; jobs are charged on whichever is larger,
#    the core fraction or the memory fraction, so --mem is always set
#  - no "-C cpu" constraint, and no iris-style account auto-detection
#  - Cray PE machine since the 2025 RHEL 9 upgrade: cray-mpich launched
#    with srun (there is no mpirun), see platform/env/env.DELTA_CPU
#
# LAYOUT EXPECTED:
#  <input-root>/batch-XXX/cgyro/input-YYY/input.cgyro
#
# USAGE:
#  ./job_scripts/submit_cgyro_delta.sh [options]
#  For example:
#  ./job_scripts/submit_cgyro_delta.sh -p ./cgyro_inputs -account bbxx-delta-cpu -s
#
#  Run without -s to print diagnostics + the first batch file only
#  (dry run); add -s to actually submit.
#------------------------------------------------------------------

#=============================================================
# Defaults (mirrors gacode_qsub defaults where applicable)
#
INPUT_ROOT=./cgyro_inputs
CODE=cgyro
PARTITION=cpu          # Delta CPU partition
WALLTIME=0:15:00       # 15 minute wallclock
SUB_FLAG=0
SKIP_DONE=0
MAX_JOBS=0             # 0 = no limit
SLEEP_SEC=0            # throttle between sbatch calls
REPO=
nmpi=16
nomp=1
ENV_SCRIPT=
EXCLUSIVE=0

# Memory request.  Precedence is -mem then -mem-per-cpu, as in gacode_qsub.
MEMPERNODE=
MEMPERCPU=

# Delta CPU node: 256 GB / 128 cores = 2 GB/core.  Slurm's usable
# RealMemory sits a little below 256 GB, so ask for slightly less than
# 2 GB/core: that keeps the memory fraction under the core fraction (so
# the job is charged on cores) and stays schedulable on a full node.
# Verify with: scontrol show node <cn###> | grep RealMemory
MEM_PER_CORE_MB=1900

# Partition wallclock ceilings (informational check only)
MAX_WALL_cpu=48:00:00
MAX_WALL_cpu_interactive=01:00:00
MAX_WALL_cpu_preempt=48:00:00

usage () {
  cat <<'EOF'

Usage:   submit_cgyro_delta.sh [options]

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
         Partition cpu allows up to 48:00:00, cpu-interactive 01:00:00,
         cpu-preempt 48:00:00.

         -queue <name>  (alias: -partition)
         Slurm partition: cpu, cpu-interactive, cpu-preempt.
         [default: cpu]

         -account <name>  (alias: -repo)
         Delta account/allocation, e.g. bbxx-delta-cpu.  Taken from
         $SLURM_ACCOUNT or from sacctmgr if not given.

         -mem <n>[unit]
         Memory required per node.  Default unit MB; Slurm accepts
         [K|M|G|T] with no space before the unit.

         -mem-per-cpu <n>[unit]
         Memory required per cpu.  Ignored if -mem is also given.
         [default: 1900M, just under the node's 2 GB/core ratio]

         -exclusive
         Request whole nodes (--exclusive --mem=0) instead of sharing.

         -env <file>
         Extra file to source inside each batch job.
         [default: $GACODE_ROOT/platform/env/env.$GACODE_PLATFORM]

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

  -queue|-partition) shift ; PARTITION=$1 ;;

  -account|-repo) shift ; REPO=$1 ;;

  -mem) shift ; MEMPERNODE=$1 ;;

  -mem-per-cpu) shift ; MEMPERCPU=$1 ;;

  -exclusive) EXCLUSIVE=1 ;;

  -env) shift ; ENV_SCRIPT=$1 ;;

  -skip-done) SKIP_DONE=1 ;;

  -max) shift ; MAX_JOBS=$1 ;;

  -sleep) shift ; SLEEP_SEC=$1 ;;

  -h|-help|--help) usage ;;

  *) echo "ERROR (submit_cgyro_delta.sh): Incorrect syntax: $1" ; exit 1 ;;

  esac
  shift
done

if [[ ! -d $INPUT_ROOT ]] ; then
   echo "ERROR: (submit_cgyro_delta.sh) Input root $INPUT_ROOT not found."
   exit 1
fi

# Absolute path to the simulation root (SIMROOT in gacode_qsub terms)
SIMROOT=$(cd "$INPUT_ROOT" && pwd)

#=============================================================
# Partition sanity
#
case "$PARTITION" in
   cpu)             MAXWALL=$MAX_WALL_cpu ;;
   cpu-interactive) MAXWALL=$MAX_WALL_cpu_interactive ;;
   cpu-preempt)     MAXWALL=$MAX_WALL_cpu_preempt ;;
   *) echo "WARNING: (submit_cgyro_delta.sh) Unrecognised partition '$PARTITION'."
      echo "         Known CPU partitions: cpu, cpu-interactive, cpu-preempt."
      MAXWALL= ;;
esac

#=============================================================
# Account (allocation) resolution -- Delta has no iris/showusage
#
if [[ -z $REPO ]] ; then
   if [[ -n ${SLURM_ACCOUNT:-} ]] ; then
      REPO=$SLURM_ACCOUNT
   elif hash sacctmgr 2> /dev/null ; then
      # Prefer a CPU allocation; fall back to the first association.
      REPO=$(sacctmgr -nP show assoc user=$USER format=account 2> /dev/null \
             | sort -u | grep -m1 'delta-cpu')
      if [[ -z $REPO ]] ; then
         REPO=$(sacctmgr -nP show assoc user=$USER format=account 2> /dev/null \
                | sort -u | head -1)
      fi
   fi
fi

if [[ -z $REPO ]] ; then
   if [[ $SUB_FLAG -eq 1 ]] ; then
      echo "ERROR: (submit_cgyro_delta.sh) No Delta account found."
      echo "       Pass one with -account <name> (e.g. bbxx-delta-cpu), or set"
      echo "       SLURM_ACCOUNT.  'accounts' lists the allocations you can use."
      exit 1
   fi
   REPO=null_account
   echo "WARNING: No Delta account found; using '$REPO' for this dry run."
fi

#=============================================================
# Environment file sourced inside each job
#
if [[ -z $ENV_SCRIPT ]] ; then
   if [[ -n ${GACODE_ROOT:-} && -n ${GACODE_PLATFORM:-} \
         && -f $GACODE_ROOT/platform/env/env.$GACODE_PLATFORM ]] ; then
      ENV_SCRIPT=$GACODE_ROOT/platform/env/env.$GACODE_PLATFORM
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
   # Fall back to Delta CPU node values so the script can be
   # inspected off-platform.
   echo "WARNING: GACODE_ROOT/GACODE_PLATFORM unset; assuming Delta CPU."
   IDENTITY="NCSA Delta CPU"
   CORES_PER_NODE=128
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
# Node sharing (the Delta default) vs whole nodes
#
cores_per_job=$((nmpi * nomp))
cores_per_node_used=$((mpinode * nomp))

if [[ $cores_per_node_used -gt $CORES_PER_NODE ]] ; then
   echo "ERROR: (submit_cgyro_delta.sh) $cores_per_node_used cores/node requested,"
   echo "       but a Delta CPU node has only $CORES_PER_NODE.  Reduce -n/-nomp."
   exit 1
fi

if [[ $EXCLUSIVE -eq 1 ]] ; then
   # Whole-node allocation: take all of the node's memory.
   MEMPERNODE=
   MEMPERCPU=
elif [[ -z $MEMPERNODE && -z $MEMPERCPU ]] ; then
   MEMPERCPU=${MEM_PER_CORE_MB}M
fi

#=============================================================
# Collect every input.cgyro under the input root
#
INPUT_FILES=()
while IFS= read -r input_path ; do
   INPUT_FILES+=("$input_path")
done < <(find "$SIMROOT" -type f -name "input.$CODE" | sort)

if [[ ${#INPUT_FILES[@]} -eq 0 ]] ; then
   echo "ERROR: (submit_cgyro_delta.sh) No input.$CODE found under $SIMROOT"
   exit 1
fi

echo "INFO: (submit_cgyro_delta.sh) Job layout diagnostics"
echo
echo "       identity: $IDENTITY"
echo " cores per node: $CORES_PER_NODE"
echo " numas per node: $NUMAS_PER_NODE"
echo
echo " simulation root         : $SIMROOT"
echo " input.$CODE files found : ${#INPUT_FILES[@]}"
echo " account                 : $REPO"
echo " partition               : $PARTITION${MAXWALL:+ (max $MAXWALL)}"
echo " wallclock               : $WALLTIME"
echo " cores requested / job   : $cores_requested"
echo " cores used / job        : $cores_used"
echo " total MPI tasks         : $nmpi"
echo " MPI tasks/node          : $mpinode"
echo " OpenMP threads/MPI task : $nomp"
echo " nodes / job             : $nodes"
if [[ $EXCLUSIVE -eq 1 ]] ; then
   echo " node allocation         : exclusive (--exclusive --mem=0)"
else
   echo " node allocation         : shared (Delta default)"
fi
if [[ -n $MEMPERNODE ]] ; then
   echo " memory per node         : $MEMPERNODE"
elif [[ -n $MEMPERCPU ]] ; then
   echo " memory per cpu          : $MEMPERCPU"
fi
if [[ -n $ENV_SCRIPT ]] ; then
   echo " env sourced in job      : $ENV_SCRIPT"
else
   echo " env sourced in job      : none (pass -env, or set GACODE_ROOT/GACODE_PLATFORM)"
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

   # ---- Batch file, generated as in platform/qsub/qsub.DELTA_CPU ----
   echo "#!/bin/bash -l" > $bfile
   echo "#SBATCH -J $JOBNAME" >> $bfile
   echo "#SBATCH -A $REPO" >> $bfile
   echo "#SBATCH -o $SIMDIR/batch.out" >> $bfile
   echo "#SBATCH -e $SIMDIR/batch.err" >> $bfile
   echo "#SBATCH -p $PARTITION" >> $bfile
   echo "#SBATCH -t $WALLTIME" >> $bfile
   echo "#SBATCH -N $nodes" >> $bfile
   echo "#SBATCH -n $nmpi" >> $bfile
   echo "#SBATCH -c $nomp" >> $bfile
   if [[ $EXCLUSIVE -eq 1 ]] ; then
      echo "#SBATCH --exclusive" >> $bfile
      echo "#SBATCH --mem=0" >> $bfile
   elif [[ -n $MEMPERNODE ]] ; then
      echo "#SBATCH --mem=$MEMPERNODE" >> $bfile
   elif [[ -n $MEMPERCPU ]] ; then
      echo "#SBATCH --mem-per-cpu=$MEMPERCPU" >> $bfile
   fi
   if [[ -n $ENV_SCRIPT ]] ; then
      echo "source $ENV_SCRIPT" >> $bfile
   fi
   if [[ -n ${GACODE_ROOT:-} ]] ; then
      # Make the job self-contained: modules above, then gacode on PATH.
      echo "export GACODE_ROOT=$GACODE_ROOT" >> $bfile
      echo "export GACODE_PLATFORM=${GACODE_PLATFORM:-DELTA_CPU}" >> $bfile
      echo "source \$GACODE_ROOT/shared/bin/gacode_setup" >> $bfile
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
         echo "ERROR: (submit_cgyro_delta.sh) No sbatch command recognized."
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
