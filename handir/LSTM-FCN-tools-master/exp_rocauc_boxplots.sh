# Bash script for experimenting with ROC curves / ROCAUC numbers


# Constants and lists
epochs=2000 # 50
seconds=240 # 30
prefix=("Explosives") # "Seawater" "Seawater_4cats" "Explosives" "Explosives_3cats" 
dataset_index=(10) # 8 9 10 11  
split=(0.3) # 0.3 0.3 0.5 0.3 #### Try explosives at 0.3

# Constants and lists for the 4-model loop specifically
## Seawater
# bats=(128 4 4 8)
# lcn=(128 0 4 8)
## Seawater_4cats
### bats=(128 4 8 16)
### lcn=(128 0 128 128)
## Explosives
bats=(128 4 4 4)
lcn=(128 0 4 16)
## Explosives_3cats
### bats=(128 4 16 4)
### lcn=(128 0 16 16)


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
    mkdir ~/results/"$i"/report
    mkdir ~/results/"$i"/conf_mat
done


# Loop over the elements of array (or any array of length array)
for ((k=0; k<${#prefix[@]}; ++k)); do

# Loop over everything five times
    for m in 1 2 3 4 5; do
    # for m in 1 2 3; do

        rm data/${prefix[k]}*

        python preprocessor.py -d unedited_data/${prefix[k]}.csv -p ${prefix[k]} -c ${split[k]}  
        ##### python generator.py -d ${prefix[k]} -r 5 -b 2 -k 1 -i _TRAIN -o _EXP_TRAIN #####

        python pca_svm.py -d ${dataset_index[k]} -p ${prefix[k]} -g 0
        mv weights/${prefix[k]}_PCASVM_report.txt ~/results/${prefix[k]}/report/${prefix[k]}_PCASVM_report_$m.txt

        python 1knn_dtw.py -d ${dataset_index[k]} -p ${prefix[k]} -g 0
        mv weights/${prefix[k]}_1knn_dtw_report.txt ~/results/${prefix[k]}/report/${prefix[k]}_1knn_dtw_report_$m.txt

        python lda.py -d ${dataset_index[k]} -p ${prefix[k]} -g 0
        mv weights/${prefix[k]}_LDA_micro_macro.csv ~/results/${prefix[k]}/ROCs/${prefix[k]}_LDA_micro_macro_$m.csv
        mv weights/${prefix[k]}_LDA_report.txt ~/results/${prefix[k]}/report/${prefix[k]}_LDA_report_$m.txt


        # Loop over each of the four models
        for i in 0 1 2 3; do
            python trainer.py -d ${dataset_index[k]} -m $i -r 1 -p ${prefix[k]} -e $epochs -b ${bats[i]} -l ${lcn[i]}
            sleep $seconds
            mv weights/${prefix[k]}_${i}_micro_macro.csv ~/results/${prefix[k]}/ROCs/${prefix[k]}_${i}_micro_macro_$m.csv
            mv weights/${prefix[k]}_${i}_report.txt ~/results/${prefix[k]}/report/${prefix[k]}_${i}_report_$m.txt
            mv weights/${prefix[k]}_${i}_conf_mat.csv ~/results/${prefix[k]}/conf_mat/${prefix[k]}_${i}_conf_mat_$m.csv
        done

        mv weights/ROCAUCs.csv ~/results/${prefix[k]}/ROCAUC/ROCAUCs_$m.csv

    done

    mv figures/* ~/results/${prefix[k]}/figures/
    mv weights/${prefix[k]}_*_parameters.txt ~/results/${prefix[k]}/parameters/

done
