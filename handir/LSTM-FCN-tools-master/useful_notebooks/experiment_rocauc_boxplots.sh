# Bash script for experimenting with ROC curves / ROCAUC numbers

# Create a place to put generated files
mkdir ~/results/Seawater/
mkdir ~/results/Seawater/rep_ROC
mkdir ~/results/Seawater/ROCAUC
mkdir ~/results/Seawater/figures
mkdir ~/results/Seawater/parameters
mkdir ~/results/Seawater_4cats/
mkdir ~/results/Seawater_4cats/rep_ROC
mkdir ~/results/Seawater_4cats/ROCAUC
mkdir ~/results/Seawater_4cats/figures
mkdir ~/results/Seawater_4cats/parameters

# Make sure ROCAUC.csv isn't already in directory
rm weights/ROCAUCs.csv

# Constants and lists
epochs=500
seconds=60
array=("Seawater" "Seawater_4cats")
array2=(4 7)
array3=(128 0 128 128) # array3=(128 0 64 64)

# Loop over the elements of array (or any array of length array)
for ((k=0; k<${#array[@]}; ++k)); do

# Loop over everything
    for m in 1 2 3; do

        rm data/${array[k]}*

        python preprocessor.py -d unedited_data/${array[k]}.csv -p ${array[k]} -c 0.2
        python generator.py -d ${array[k]} -r 5 -b 2 -k 1 -i _TRAIN -o _EXP_TRAIN
        python lda.py -d ${array2[k]} -p ${array[k]}
        mv weights/${array[k]}_LDA_micro_macro.csv ~/results/${array[k]}/rep_ROC

        for i in 0 1 2 3; do
            python trainer.py -d ${array2[k]} -m $i -r 1 -p ${array[k]} -e $epochs -b 32 -l ${array3[i]}
            sleep $seconds
            mv weights/${array[k]}_${i}_micro_macro.csv ~/results/${array[k]}/rep_ROC
        done

        mv weights/ROCAUCs.csv ~/results/${array[k]}/ROCAUC
        cp ~/results/${array[k]}/ROCAUC/ROCAUCs.csv ~/results/${array[k]}/ROCAUC/ROCAUCs_$m.csv
        rm ~/results/${array[k]}/ROCAUC/ROCAUCs.csv

    done

    mv figures/* ~/results/${array[k]}/figures/
    mv weights/${array[k]}_*_parameters.txt ~/results/${array[k]}/parameters/

done
