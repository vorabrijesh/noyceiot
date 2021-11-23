#!/bin/bash

# usage: ./scripting-auto.sh <abs_path_to_art_lib> <abs_path_to_venv> <train - 0/1>

art_nncf_venv_path=$1
train_flag=$2
attack_flag=$3
json_path=$4

# echo "${art_library_path} ${art_venv_path}"

# sleep 400
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$art_library_path #$LD_LIBRARY_PATH:/home/shafiz/ART/adversarial-robustness-toolbox/

N_PER_CLASS_TRAINING_SAMPLES=5000
N_BATCH_SIZE=128
N_EPOCHS=10

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

ATTACK_NAME=(FastGradientMethod Deepfool UniversalPerturbation BasicIterativeMethod ElasticNet Wasserstein AdversarialPatch AutoProjectedGradientDescent ShadowAttack NewtonFool TargetedUniversalPerturbation CarliniWagner)

DATASET_INDEX=0
MODEL_INDEX_START=0
MODEL_INDEX_END=3
ATTACK_INDEX_START=2
ATTACK_INDEX_END=4

CTIME="`date +%b-%d-%Y-%H-%M-%p`" 

for (( MODEL_INDEX=${MODEL_INDEX_START}; MODEL_INDEX<${MODEL_INDEX_END}+1; MODEL_INDEX++ ))
do
    PRINT_OUTPUT_FILE="output-${DATASET[$DATASET_INDEX]}-${MODEL_NAME[$MODEL_INDEX]}-${CTIME}.log"
    echo $PRINT_OUTPUT_FILE
    echo "Dataset: ${DATASET[$DATASET_INDEX]}, Fed_training: ${N_TRAINING_SAMPLES}, Fed_testing: ${N_TESTING_SAMPLES}, Fed_adv: ${N_ADV_SAMPLES}, Model: ${MODEL_NAME[$MODEL_INDEX]} ***** Model Start ***** " > $PRINT_OUTPUT_FILE

    CLASSIFIER_FILE_PREFIX="classifier-${MODEL_NAME[$MODEL_INDEX]}-${DATASET[$DATASET_INDEX]}-on-${N_TRAINING_SAMPLES}"
     
    if [ $train_flag -eq 1 ]; then
        source "${art_nncf_venv_path}bin/activate" #/home/shafiz/ART/ART-venv-default/bin/activate
        python3 smh-train-classifier-a2a.py $N_TRAINING_SAMPLES $N_BATCH_SIZE $N_EPOCHS $CLASSIFIER_FILE_PREFIX ${MODEL_NAME[$MODEL_INDEX]} >> $PRINT_OUTPUT_FILE
        deactivate
    fi 

    for (( ATTACK_INDEX=${ATTACK_INDEX_START}; ATTACK_INDEX<${ATTACK_INDEX_END}+1; ATTACK_INDEX++ ))
    do
        echo -e "\n\nFed_adv: ${N_ADV_SAMPLES}, Attack: ${ATTACK_NAME[$ATTACK_INDEX]} ***** Attack Start ***** " >> $PRINT_OUTPUT_FILE
        if [ $attack_flag -eq 1 ]; then
            source "${art_nncf_venv_path}bin/activate"
            python3 smh-subset-of-test.py $N_PER_CLASS_TESTING_SAMPLES $N_CLASSES ${DATASET[$DATASET_INDEX]} >> $PRINT_OUTPUT_FILE
            python3 smh-attack-and-adv-examples.py $CLASSIFIER_FILE_PREFIX ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_TESTING_SAMPLES >> $PRINT_OUTPUT_FILE
            deactivate
        fi 

        source "${art_nncf_venv_path}bin/activate"
        python3 smh-subset-of-test-adv.py $N_PER_CLASS_TESTING_SAMPLES $N_CLASSES ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_ADV_SAMPLES >> $PRINT_OUTPUT_FILE
        json_file="${json_path}${MODEL_NAME[$MODEL_INDEX]}/${MODEL_NAME_LOWER[$MODEL_INDEX]}-${DATASET[$DATASET_INDEX]}-pruning_50%_L2.json"
        echo $json_file
        python3 smh-nncf-results.py ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_ADV_SAMPLES $CLASSIFIER_FILE_PREFIX $json_file $N_BATCH_SIZE >> $PRINT_OUTPUT_FILE
        python3 smh-nncf-a2a-results.py ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_ADV_SAMPLES $CLASSIFIER_FILE_PREFIX $json_file >> $PRINT_OUTPUT_FILE
        deactivate
    done
done
echo "Done!"




