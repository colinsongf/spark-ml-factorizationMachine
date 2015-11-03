# Spark Aol-ml
Aol-ml is a generalized *Apache Spark ML* extension that currently includes *Factorization Machine*

## Requirements
* Spark 1.5.1 or above 
* Scala 2.10 or above
* Sbt 0.13.9 or above (alternatively one can try building with Maven)

## Obtaining the source code
Aol-ml is available on Stash: ``` git clone git clone https://stash.ops.aol.com/scm/adl/adl-aol-ml.git ``` <br/>

## Building/packaging the core
At the source directory (where **build.sbt** is located) run ```sbt (clean) package``` in console/shell at the source <br \>

## Assembling the software (output is called fat-jar)
Invoke ```sbt (clean) assembly``` in console/shell (requires `sbt-assembly`; automatically pulled in by sbt).

## Run the program
The program can be run in three modes
* Locally <br \>
One can run locally by invoking ```sbt 'run -l -o OUTPUT -i INPUT_DATA'``` (Do not forget the `-l` or `-runLocally` flag)
* Using `spark-submit`
Alternatively one can run on Spark using 

```
spark-submit \ 
--class "com.aol.advertising.experimental.Driver" \ 
--master "local[4]" \    ([4] signifies number of cores; optionally run on Spark cluster by specifying address here) 
TARGET_ASSEMBLY_JAR \    (usually inside target directory, if assembly/packaging was successful)
TARGET_INPUT_OPTIONS     (e.g. '-i INPUT_DATA -o OUTPUT_NAME')
```
Refer the Spark documentation for more options.
