# Bash script for experimenting with LSTM cell number

# Constants and lists
epochs=500  # 500
seconds=20 # 120
prefix=("Seawater" "Seawater_4cats" "Explosives" "Explosives_3cats") # 
dataset_index=(8 9 10 11) #  
split=(0.3 0.3 0.5 0.3) #  
model=3 # 0 2 3 (Not 1 because... FCN doesn't have a LCN parameter)
# LSTM
# bat=(128 128 128 128)
# LSTM-FCN
# bat=(4 8 4 16)
# ALSTM-FCN
bat=(8 16 4 4) #  


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



for ((k=0; k<${#prefix[@]}; ++k)); do 

    for m in 1 2 3; do

        rm data/${prefix[k]}*

        python preprocessor.py -d unedited_data/${prefix[k]}.csv -p ${prefix[k]} -c ${split[k]}

        for i in 128 64 32 16 8 4; do
            python trainer.py -d ${dataset_index[k]} -m $model -r 1 -p ${prefix[k]} -e $epochs -l $i -b ${bat[k]} # keep m constant; change l
            sleep $seconds
        done

        mv weights/ROCAUCs.csv ~/results/${prefix[k]}/ROCAUC/ROCAUCs_$m.csv
    done


    mv figures/* ~/results/${prefix[k]}/figures/
    mv weights/${prefix[k]}_*_parameters.txt ~/results/${prefix[k]}/parameters/
done