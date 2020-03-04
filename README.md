# Presum-Test-Results
The last week we tried to reproduce the results reported by Yang Liu et al., (https://arxiv.org/abs/1908.08345), on using pretrained BERT model to extractively summarize the text.<b/>  
Here is their result:<b/>     
BERTSUMEXT 43.25 20.24 39.63<b/> 
Since we are using a NV6 with single GPU we ran the training (in the same number of training step) with CNN and Daily Mail dataset but about ¼ of batch size. Still, the result is pretty impressive:<b/>  
>> ROUGE-F(1/2/3/l): 41.91/19.09/38.31 <b/>

## Steps
We got the source code and the preprocessed Porth CNNDM data set, in a set of *.pt files from the GtiHub repo https://github.com/nlpyang/PreSummmaster master branch. <b/>

We made a one-line change in ../src/models/data_loader.py to enable to process all data files: we changed line 84 in data_loader.py to<b/> 

pts = sorted(glob.glob(args.bert_data_path + '/[a-z]*.' + corpus_type + '.[0-9]*.bert.pt'))   

Since the training process will write out many checkpoint files and can be 2-3 GB size each, we add a 1TB SDD data disk to the VM and mounted as /datadrive and we cloned the code on it to avoid OS disk full. 

Since the training process can take many hours (20+?), we also set up Azure Bastion to web-FTP to the VM so we can check the progress and results anyplace and no worries about the VM network security setting.  

## Commands
The commands we used for train, validate and test are   <b/>
python3 train.py -task ext -mode train -bert_data_path ../bert_data/cnndm -ext_dropout 0.1 -model_path ../models -lr 2e-3 -visible_gpus 0 -report_every 50 -save_ checkpoint_steps 1000 -batch_size 800 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_po 512 &amp; 

python3 train.py -task ext -mode validate -batch_size 800 -test_batch_size 64 -bert_data_path ../bert_data/cnndm -log_file ../logs/val_abs_bert_cnndm -model_path  ../models/ext_bert_cnndm -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm &amp; 

python3 train.py -task ext -mode test -batch_size 800 -test_batch_size 64 -bert_data_path ../bert_data/cnndm -log_file ../logs/test_abs_bert_cnndm -model_path .. /bert_data/cnndm -log_file ../logs/test_abs_bert_cnndm -test_from ../models/ext_bert_cnndm/model_step_50000.pt -sep_optim true -use_interval true -visible_gpus 0 -max_p os 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../results &amp; 

## Results
A good indication that the training is converged is that the cross entropy (xent) reduced to 10.56 to 1.85: <b/>
[2020-02-29 05:41:43,049 INFO] Step 50/50000; xent: 10.56; lr: 0.0000001;  10 docs/s;     21 sec<b/>
[2020-03-01 00:59:48,282 INFO] Step 50000/50000; xent: 1.85; lr: 0.0000089;  22 docs/s;  26876 sec<b/>

