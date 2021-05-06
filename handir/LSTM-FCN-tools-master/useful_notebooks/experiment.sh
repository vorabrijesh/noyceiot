# Bash script for experimenting

# mkdir ~/results/Seawater/
# mkdir ~/results/Seawater/ROCAUC
# mkdir ~/results/Seawater/figures
# mkdir ~/results/Seawater/parameters
# mkdir ~/results/Seawater_4cats/
# mkdir ~/results/Seawater_4cats/ROCAUC
# mkdir ~/results/Seawater_4cats/figures
# mkdir ~/results/Seawater_4cats/parameters

epochs=500
seconds=30
array=("Seawater" "Seawater_4cats")
array2=(4 7)

for ((k=0; k<${#array[@]}; ++k)); do
    for m in 1 2 3; do

        rm data/${array[k]}*

        python preprocessor.py -d unedited_data/${array[k]}.csv -p ${array[k]} -c 0.2
        python generator.py -d ${array[k]} -r 5 -b 2 -k 1 -i _TRAIN -o _EXP_TRAIN
        python lda.py -d ${array2[k]} -p ${array[k]}

        for i in 0 1 2 3; do
            python trainer.py -d ${array2[k]} -m $i -r 1 -p ${array[k]} -e $epochs -b 16
            sleep $seconds
        done

        mv weights/ROCAUCs.csv ~/results/${array[k]}/ROCAUC
        cp ~/results/${array[k]}/ROCAUC/ROCAUCs.csv ~/results/${array[k]}/ROCAUC/ROCAUCs_$m.csv
        rm ~/results/${array[k]}/ROCAUC/ROCAUCs.csv
    done


    mv figures/* ~/results/${array[k]}/figures/
    mv weights/${array[k]}_*_parameters.txt ~/results/${array[k]}/parameters/
done