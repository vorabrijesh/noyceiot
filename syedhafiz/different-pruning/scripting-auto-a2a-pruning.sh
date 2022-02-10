#!/bin/bash

# usage: ./scripting-auto-a2a-pruning.sh <abs_path_to_venv_with_art_nncf> <train - 0/1> <abs_path_to_pruning_configs>

art_nncf_venv_path=$1
train_flag=$2
attack_flag=$3
json_path=$4

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$art_library_path

N_PER_CLASS_TRAINING_SAMPLES=5000
N_BATCH_SIZE=128
N_EPOCHS=20

N_PER_CLASS_TESTING_SAMPLES=1000

N_PER_CLASS_ADV_SAMPLES=$N_PER_CLASS_TESTING_SAMPLES

N_CLASSES=10
N_TRAINING_SAMPLES=$(($N_PER_CLASS_TRAINING_SAMPLES*$N_CLASSES))
N_TESTING_SAMPLES=$(($N_PER_CLASS_TESTING_SAMPLES*$N_CLASSES))
N_ADV_SAMPLES=$(($N_PER_CLASS_ADV_SAMPLES*$N_CLASSES))
TRT_INPUT_1D=32

DATASET=(cifar10 imagenet)
MODEL_NAME=(MobileNet DenseNet121 VGG19 ResNet50)
MODEL_NAME_LOWER=(mobilenet densenet121 vgg19 resnet50)
PRUNE_TARGET=(0.30 0.10)
FILTER_IMPORTANCE=(L1 L2 geometric_median)

ATTACK_NAME=(FastGradientMethod Deepfool UniversalPerturbation BasicIterativeMethod ElasticNet CarliniWagner Wasserstein AdversarialPatch AutoProjectedGradientDescent ShadowAttack NewtonFool TargetedUniversalPerturbation CarliniWagner)

DATASET_INDEX=0

MODEL_INDEX_START=0
MODEL_INDEX_END=3

ATTACK_INDEX_START=1
ATTACK_INDEX_END=3

PRUNE_TARGET_INDEX_START=0
PRUNE_TARGET_INDEX_END=1

FILTER_IMPORTANCE_INDEX_START=0
FILTER_IMPORTANCE_INDEX_END=2

CTIME="`date +%b-%d-%Y-%H-%M-%p`" 

for (( MODEL_INDEX=${MODEL_INDEX_START}; MODEL_INDEX<${MODEL_INDEX_END}+1; MODEL_INDEX++ ))
do
    PRINT_OUTPUT_FILE="output-${DATASET[$DATASET_INDEX]}-${MODEL_NAME[$MODEL_INDEX]}-${CTIME}.log"
    echo $PRINT_OUTPUT_FILE
    echo "Dataset: ${DATASET[$DATASET_INDEX]}, Fed_training: ${N_TRAINING_SAMPLES}, Fed_testing: ${N_TESTING_SAMPLES}, Fed_adv: ${N_ADV_SAMPLES}, Model: ${MODEL_NAME[$MODEL_INDEX]} ***** Model Start ***** " > $PRINT_OUTPUT_FILE

    CLASSIFIER_FILE_PREFIX="classifier-${MODEL_NAME[$MODEL_INDEX]}-${DATASET[$DATASET_INDEX]}-on-${N_TRAINING_SAMPLES}"
     
    if [ $train_flag -eq 1 ]; then
        source "${art_nncf_venv_path}bin/activate"
        python3 smh-train-classifier-a2a.py $N_TRAINING_SAMPLES $N_BATCH_SIZE $N_EPOCHS $CLASSIFIER_FILE_PREFIX ${MODEL_NAME[$MODEL_INDEX]} >> $PRINT_OUTPUT_FILE
        deactivate
    fi 

    for (( ATTACK_INDEX=${ATTACK_INDEX_START}; ATTACK_INDEX<${ATTACK_INDEX_END}+1; ATTACK_INDEX++ ))
    do
        echo -e "\n\nFed_adv: ${N_ADV_SAMPLES}, Attack: ${ATTACK_NAME[$ATTACK_INDEX]} ***** Attack Start ***** " >> $PRINT_OUTPUT_FILE
        if [ $attack_flag -eq 1 ]; then
            source "${art_nncf_venv_path}bin/activate"
            python3 smh-subset-of-test.py $N_PER_CLASS_TESTING_SAMPLES $N_CLASSES ${DATASET[$DATASET_INDEX]} >> $PRINT_OUTPUT_FILE
            python3 smh-attack-and-adv-examples.py $CLASSIFIER_FILE_PREFIX ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_TESTING_SAMPLES $N_BATCH_SIZE >> $PRINT_OUTPUT_FILE
            deactivate
        fi 

        source "${art_nncf_venv_path}bin/activate"
        python3 smh-subset-of-test-adv.py $N_PER_CLASS_TESTING_SAMPLES $N_CLASSES ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_ADV_SAMPLES >> $PRINT_OUTPUT_FILE
        for (( FILTER_IMPORTANCE_INDEX=${FILTER_IMPORTANCE_INDEX_START}; FILTER_IMPORTANCE_INDEX<${FILTER_IMPORTANCE_INDEX_END}+1; FILTER_IMPORTANCE_INDEX++ ))
        do
            echo -e "\nFILTER_IMPORTANCE_INDEX: ${FILTER_IMPORTANCE[FILTER_IMPORTANCE_INDEX]} ***** FILTER_IMPORTANCE_INDEX  Start ***** " >> $PRINT_OUTPUT_FILE
            for (( PRUNE_TARGET_INDEX=${PRUNE_TARGET_INDEX_START}; PRUNE_TARGET_INDEX<${PRUNE_TARGET_INDEX_END}+1; PRUNE_TARGET_INDEX++ ))
            do
                echo -e "\nPRUNE_TARGET_INDEX: ${PRUNE_TARGET[$PRUNE_TARGET_INDEX]} ***** PRUNE_TARGET_INDEX  Start ***** " >> $PRINT_OUTPUT_FILE
                json_file="${json_path}${MODEL_NAME[$MODEL_INDEX]}/${MODEL_NAME_LOWER[$MODEL_INDEX]}-${DATASET[$DATASET_INDEX]}-pruning_${PRUNE_TARGET[$PRUNE_TARGET_INDEX]}_${FILTER_IMPORTANCE[FILTER_IMPORTANCE_INDEX]}.json" #$((${PRUNE_TARGET[$PRUNE_TARGET_INDEX]}*100))
                echo $json_file
                python3 ../art-plus-nncf/configs/pruning/gen_json.py ../art-plus-nncf/configs/pruning/pruning_base.json ${MODEL_NAME[$MODEL_INDEX]} 1 1,32,32,3 $N_BATCH_SIZE $N_EPOCHS  ${PRUNE_TARGET[$PRUNE_TARGET_INDEX]} ${FILTER_IMPORTANCE[$FILTER_IMPORTANCE_INDEX]} $json_file
                python3 smh-nncf-results.py ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_ADV_SAMPLES $CLASSIFIER_FILE_PREFIX $json_file $N_BATCH_SIZE >> $PRINT_OUTPUT_FILE
                python3 smh-nncf-a2a-results.py ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_ADV_SAMPLES $CLASSIFIER_FILE_PREFIX $json_file $N_BATCH_SIZE >> $PRINT_OUTPUT_FILE
            done
        done
        deactivate
    done
done
echo "Done!"