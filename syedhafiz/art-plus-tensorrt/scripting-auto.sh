#!/bin/bash

# usage: ./scripting-auto.sh <abs_path_to_art_lib> <abs_path_to_venv> <train - 0/1>

art_path=$1
venv_path=$2
train_flag=$3

# echo "${art_path} ${venv_path}"

# sleep 400
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$art_path #$LD_LIBRARY_PATH:/home/shafiz/ART/adversarial-robustness-toolbox/

N_PER_CLASS_TRAINING_SAMPLES=5000
N_BATCH_SIZE=128
N_EPOCHS=20

N_PER_CLASS_TESTING_SAMPLES=800

N_PER_CLASS_ADV_SAMPLES=$N_PER_CLASS_TESTING_SAMPLES

N_CLASSES=10
N_TRAINING_SAMPLES=$(($N_PER_CLASS_TRAINING_SAMPLES*$N_CLASSES))
N_TESTING_SAMPLES=$(($N_PER_CLASS_TESTING_SAMPLES*$N_CLASSES))
N_ADV_SAMPLES=$(($N_PER_CLASS_ADV_SAMPLES*$N_CLASSES))
TRT_INPUT_1D=32

DATASET=(cifar10 imagenet)
MODEL_NAME=(VGG19 ResNet50 MobileNet DenseNet121)

ATTACK_NAME=(CarliniWagner Deepfool FastGradientMethod ElasticNet Wasserstein AdversarialPatch AutoProjectedGradientDescent ShadowAttack UniversalPerturbation BasicIterativeMethod)

DATASET_INDEX=0
MODEL_INDEX_START=3
MODEL_INDEX_END=3
ATTACK_INDEX_START=2
ATTACK_INDEX_END=2


CTIME="`date +%b-%d-%Y-%H-%M-%p`" 

for (( MODEL_INDEX=${MODEL_INDEX_START}; MODEL_INDEX<${MODEL_INDEX_END}+1; MODEL_INDEX++ ))
do
    PRINT_OUTPUT_FILE="output-${DATASET[$DATASET_INDEX]}-${MODEL_NAME[$MODEL_INDEX]}-${CTIME}.log"
    echo $PRINT_OUTPUT_FILE
    echo "Dataset: ${DATASET[$DATASET_INDEX]}, Fed_training: ${N_TRAINING_SAMPLES}, Fed_testing: ${N_TESTING_SAMPLES}, Fed_adv: ${N_ADV_SAMPLES}, Model: ${MODEL_NAME[$MODEL_INDEX]} ***** Model Start ***** " > $PRINT_OUTPUT_FILE

    CLASSIFIER_FILE_PREFIX="classifier-${MODEL_NAME[$MODEL_INDEX]}-${DATASET[$DATASET_INDEX]}-on-${N_TRAINING_SAMPLES}"
     
    #update the path to your python-3-created virtual environment
    if [ $train_flag -eq 1 ]; then
        source "${venv_path}bin/activate" #/home/shafiz/ART/ART-venv-default/bin/activate
        python3 smh-train-classifier.py $N_TRAINING_SAMPLES $N_BATCH_SIZE $N_EPOCHS $CLASSIFIER_FILE_PREFIX ${MODEL_NAME[$MODEL_INDEX]} >> $PRINT_OUTPUT_FILE
        deactivate
    fi 

    for (( ATTACK_INDEX=${ATTACK_INDEX_START}; ATTACK_INDEX<${ATTACK_INDEX_END}+1; ATTACK_INDEX++ ))
    do
        echo "Fed_adv: ${N_ADV_SAMPLES}, Attack: ${ATTACK_NAME[$ATTACK_INDEX]} ***** Attack Start ***** " >> $PRINT_OUTPUT_FILE
        #update the path to your python-3-created virtual environment
        #source /home/shafiz/ART/ART-venv-default/bin/activate
        source "${venv_path}bin/activate"
        python3 smh-subset-of-test.py $N_PER_CLASS_TESTING_SAMPLES $N_CLASSES ${DATASET[$DATASET_INDEX]} >> $PRINT_OUTPUT_FILE
        python3 smh-attack-and-adv-examples.py $CLASSIFIER_FILE_PREFIX ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_TESTING_SAMPLES >> $PRINT_OUTPUT_FILE
        deactivate

        python3 smh-subset-of-test-adv.py $N_PER_CLASS_TESTING_SAMPLES $N_CLASSES ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_ADV_SAMPLES >> $PRINT_OUTPUT_FILE
        #python3 smh-keras-to-tensorrt.py $TRT_INPUT_1D ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_ADV_SAMPLES $CLASSIFIER_FILE_PREFIX >> $PRINT_OUTPUT_FILE
        #python3 smh-tensorrt-results.py ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_ADV_SAMPLES >> $PRINT_OUTPUT_FILE
    done
done
echo "Done!"




