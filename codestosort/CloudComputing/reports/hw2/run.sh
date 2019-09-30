#!/bin/bash
# Hadoop stream jar
STREAMJAR=/opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.1.2.jar
# input file
INPUT=access.log
# input directory
INPUT_DIR=/input
# output file
OUTPUT=result.dat
# output directory
OUTPUT_DIR=/output
# mapper file
MAPPER=./mapper.py
# reducer file
REDUCER=./reducer.py
# create input directory on hdfs
hdfs dfs -mkdir /input
# upload input file to input directory
hdfs dfs -put $INPUT $INPUT_DIR
# remove old output directory
hdfs dfs -rm -r -f $OUTPUT_DIR
# execute map-reduce with Hadoop stream jar
hadoop jar $STREAMJAR -files $MAPPER,$REDUCER -mapper $MAPPER -reducer $REDUCER -input $INPUT_DIR -output $OUTPUT_DIR
# download the output file from hdfs
hdfs dfs -cat $OUTPUT_DIR/part* $OUTPUT