#!/bin/bash

source dsivenv/bin/activate

mv probs ../probs
mv results ../results
pip install .
mv ../probs probs
mv ../results results
