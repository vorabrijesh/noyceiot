# Bash script for experimenting with ROC curves / ROCAUC numbers


# Constants and lists
epochs=10 # 1000
seconds=20 # 240
prefix=("Seawater" "Seawater_4cats" "Explosives" "Explosives_3cats")
dataset_index=(8 9 10 11)
lcn=(128 0 32 32)
split=(0.2 0.2 0.5 0.2)
bats=(128 4 4 4)


# Remove files and dirs
rm weights/ROCAUCs.csv
rm figures/*
for i in "${prefix[@]}"; do
    rm -r ~/results/"$i"/
done


# Create a place to put generated files
for i in "${prefix[@]}"; do
    mkdir ~/results/"$i"/
    mkdir ~/results/"$i"/ROCs
    mkdir ~/results/"$i"/ROCAUC
    mkdir ~/results/"$i"/figures
    mkdir ~/results/"$i"/parameters
done


# Loop over the elements of array (or any array of length array)
for ((k=0; k<${#prefix[@]}; ++k)); do

# Loop over everything three times
    for m in 1 2 3; do

        rm data/${prefix[k]}*

        python preprocessor.py -d unedited_data/${prefix[k]}.csv -p ${prefix[k]} -c ${split[k]}  
        # python generator.py -d ${prefix[k]} -r 5 -b 2 -k 1 -i _TRAIN -o _EXP_TRAIN #
        python lda.py -d ${dataset_index[k]} -p ${prefix[k]} -g 0
        mv weights/${prefix[k]}_LDA_micro_macro.csv ~/results/${prefix[k]}/ROCs/${prefix[k]}_LDA_micro_macro_$m.csv

        # Loop over each of the four models
        for i in 0 1 2 3; do
            python trainer.py -d ${dataset_index[k]} -m $i -r 1 -p ${prefix[k]} -e $epochs -b ${bats[k]} -l ${lcn[i]}
            sleep $seconds
            mv weights/${prefix[k]}_${i}_micro_macro.csv ~/results/${prefix[k]}/ROCs/${prefix[k]}_${i}_micro_macro_$m.csv
        done

        mv weights/ROCAUCs.csv ~/results/${prefix[k]}/ROCAUC/ROCAUCs_$m.csv

    done

    mv figures/* ~/results/${prefix[k]}/figures/
    mv weights/${prefix[k]}_*_parameters.txt ~/results/${prefix[k]}/parameters/

done
