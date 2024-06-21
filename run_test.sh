#!/bin/bash

mt=pubmedbert
fpath=$1

python -m run.testing.run_pel_test --text-path $1 --model-type $mt
python -m run.testing.run_pel_test --text-path $1 --model-type $mt --extra-context
python -m run.testing.run_pel_test --text-path $1 --model-type $mt --superposition
python -m run.testing.run_pel_test --text-path $1 --model-type $mt --superposition --extra-context