# MS_CPF_KD

This project work combines the work of CPF student and MultiScale KD in GNNs papers. The references have been provided in the presentation of this project.


### SETUP
Setup conda environment with enviornment.yml file
```
conda env create -file environment.yml
```

### Training scripts:
```
python train_teacher.py --teacher GAT  --gpu 0 --dataset cora
python train_student.py --teacher GAT --student PLP --ptype ind --gpu 0 --dataset cora
```
The teachers will be trained in models/ directory, and afterwards you can train your student from the teachers that are there in the models/ directory.  
  
**NOTE:** If you get an error, you've probably forgot to delete the models/ directory :)


**This code should only be used for academic purposes.**
