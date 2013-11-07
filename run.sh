FILE_PAHT=$1
OBJE_NAME=$2
LINES=`wc -l $1| awk '{print $1}'`
echo '############################################'
echo "# Input File Path : $1"
echo "# Object Name     : $2"
echo '############################################'
###########CLEAN##########
echo '\033[31m Cleaning Data'
/usr/local/bin/python2.7 FreqBase.py clean $FILE_PAHT > tmp
echo '\033[31m Finish Cleaning'
echo '        |'
###########SORT############
echo '\033[31m Sorting Data'
awk ' { printf("%d\t%s\n",length,$0)}' tmp  | sort -n -k 1,1 | cut -f 2- > tmp_sorted
rm tmp
echo  '\033[31m Finish Sorting'
echo '        |'

###########SPLIT############
echo  '\033[31m Splitting Data'
TOTAL_LINES=`wc -l tmp_sorted | awk '{print $1}'`
SPLIT_LINE=`expr $TOTAL_LINES \* 2 / 3`
sed -n '1,'$SPLIT_LINE'p' tmp_sorted > train_data.txt
sed -n ''$SPLIT_LINE','$TOTAL_LINES'p' tmp_sorted > test_data.txt
rm tmp_sorted
echo  '\033[31m Finish Splitting Data'
echo '        |'

##########TRAIN#############
echo  '\033[31m Start Training'
/usr/local/bin/python2.7 FreqBase.py gen_model train_data.txt > model.txt
echo  '\033[31m Finish Training'
echo '        |'

##########TEST#############
echo  '\033[31m Start Testing'
/usr/local/bin/python2.7 FreqBase.py classify xiaomi/precision_test.txt $OBJE_NAME model.txt > rst.txt
echo  '\033[31m Finish Testing'
echo '        |'

##########CLEAN#############
echo  '\033[31m Start Cleaning\033[0m'
rm model.txt
echo '  remove model file'
rm train_data.txt
echo '  remove train file'
rm test_data.txt
echo '  remove test file'
echo  '\033[31m Finish Cleaning\033[0m'


##########SHOW##############
echo '############################################'
echo '#\033[34m OUTPUT FILE: rst.txt\033[0m'
echo '############################################'
