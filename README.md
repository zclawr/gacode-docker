# gacode-docker

This repository contains scripts for running Gacode simulations (TGLF or CGYRO) in Kubernetes jobs designed for the Nautilus cluster, which uses the [Zihao's Toolbox](https://github.com/Rose-STL-Lab/Zihao-s-Toolbox) repository for S3 utility functions and job scheduling. 

## Running a simulation job

In order to run a simulation job, make sure that the ```username``` field in ```config/kube.yaml``` contains your username on the Nautilus cluster. 

Also create a ```.env``` file in the root directory containing S3 bucket specifics and any required credentials for a private bucket. The bucket name and endpoint URL fields are required:
```
S3_BUCKET_NAME= ...
S3_ENDPOINT_URL= ...
# Only necessary if bucket is private
AWS_ACCESS_KEY_ID= ...
AWS_SECRET_ACCESS_KEY= ...
```

The code flow for the simulation job is as follows:
- Download inputs from specified S3 path, from an S3 bucket specified in ```.env```
- Run the inputs through TGLF or CGYRO, depending on which simulator is specified in the args for ```run_simulation_job.sh```
- Upload outputs back to S3 at the same path that they were downloaded from. This respects any pre-existing directory structure at the S3 path.

To run the simulation job, run the following command from the root directory:
```
bash ./run_simulation_job.sh <tglf OR cgyro> <s3path>
```

Where ```s3path``` is a path in the S3 bucket to inputs for simulation. Currently, the Docker image that runs the simulation jobs expects the following directory structure (consider ```batch-001``` to stored in the root directory of the bucket, although this doesn't have to be the case): 
```
- batch-001
    - tglf
        - input-001
            - input.tglf
        - input-002
            - input.tglf
        ...
    - cgyro
        - input-001
            - input.cgyro
        - input-002
            - input.cgyro
        ...
```

In order to run all of these inputs, then, you could execute the following from the root directory:
```
bash ./run_simulation_job.sh tglf batch-001/tglf/
bash ./run_simulation_job.sh cgyro batch-001/cgyro/
```

This will make two separate Nautilus jobs, one of which will run TGLF, and therefore will not request any GPU resources, and one of which will run CGYRO, and therefore will request GPU resources. Each of these jobs will simulate all of the inputs in their respective directories (```~/tglf/``` or ```~/cgyro/```).

Notice that there is no leading slash on the S3 path, but there is a trailing slash. This is important for the S3 utils pathing.