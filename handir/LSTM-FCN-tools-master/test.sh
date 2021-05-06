# for i in {1..25}; do
#     echo $i
# done

prefix=("Seawater_all" "Seawater_all4cats" "Explosives" "Explosives_3cats")

for k in "${prefix[@]}"; do
    source parameter_files/"$k".txt
    for i in 0 1 2 3; do
        echo "${bats[i]}"
    done
done