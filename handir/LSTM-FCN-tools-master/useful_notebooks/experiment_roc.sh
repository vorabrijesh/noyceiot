# Bash script for experimenting with ROC curves / ROCAUC numbers

# Create a place to put generated files
# ...

# Make sure ROCAUC.csv isn't already in directory
rm weights/ROCAUCs.csv

# Constants and lists
epochs=500
seconds=60
array=("Seawater" "Seawater_4cats")
array2=(4 7)
array3=(128 0 64 64) # optimal LSTM cell numbers

# Loop over the elements of array (or any array of length array)
for ((k=0; k<${#array[@]}; ++k)); do

# Loop over 
    # for m in 1 2 3; do  ### If you need boxplots made of ROCAUC

        rm data/${array[k]}*

        python preprocessor.py -d unedited_data/${array[k]}.csv -p ${array[k]} -c 0.2
        python generator.py -d ${array[k]} -r 5 -b 2 -k 1 -i _TRAIN -o _EXP_TRAIN
        python lda.py -d ${array2[k]} -p ${array[k]}

        for i in 0 1 2 3; do
            python trainer.py -d ${array2[k]} -m $i -r 1 -p ${array[k]} -e $epochs -b 32 -l ${array3[i]}
            sleep $seconds
        done

        # mv weights/ROCAUCs.csv ~/results/${array[k]}/ROCAUC
        # cp ~/results/${array[k]}/ROCAUC/ROCAUCs.csv ~/results/${array[k]}/ROCAUC/ROCAUCs_$m.csv
        # rm ~/results/${array[k]}/ROCAUC/ROCAUCs.csv

    # done                  ### If you need boxplots made of ROCAUC


    # mv figures/* ~/results/${array[k]}/figures/
    # mv weights/${array[k]}_*_parameters.txt ~/results/${array[k]}/parameters/
done