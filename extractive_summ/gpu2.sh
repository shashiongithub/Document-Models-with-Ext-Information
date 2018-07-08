# source ~/.bash_rc
# source ~/.bash_profile
# activate-tensorflow
# export TMP=/tmp/<my-temp-dir>

## Title + Captions
# python document_summarizer_gpu2.py --tmp_directory /tmp/<my-temp-dir> --max_image_length 10 --train_dir <my-training-dir> > train.log

# python document_summarizer_gpu2.py --max_title_length 1 --max_image_length 10 --train_dir <my-training-dir> > train.log

python document_summarizer_gpu2.py --max_title_length 1 --max_image_length 10 --train_dir <my-training-dir> --model_to_load 8 --exp_mode test > test.log

## First sentence baseline
# python document_summarizer_gpu2.py --max_image_length 10 --max_firstsentences_length 1 --train_dir <my-training-dir> > train.fs.log

