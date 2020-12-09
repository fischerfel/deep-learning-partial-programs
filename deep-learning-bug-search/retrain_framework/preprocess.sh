#!bin/sh

if [ ! -d "output" ]; then
	mkdir output
fi

if [ ! -d "retrain/added_data" ]; then
	mkdir retrain/added_data
fi

if [ ! -d "retrain/saved_model" ]; then
	mkdir retrain/saved_model
fi

rm graphlist
for file in `find ../../data -name '*.acfgs'`
do
	echo $file >> graphlist
done
