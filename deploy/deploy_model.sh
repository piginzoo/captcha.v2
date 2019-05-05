#!/bin/bash
CONTAINER_ID=`docker ps|awk 'END{print $NF}'`
MODEL_DIR=/home/captcha/model
BANKS=("jianhang" "renfa" "nonghang" "icbc" "crr" "psbc" "boc");
HOME=/app/captcha
BACKUP_DIR=backup_`date +"%Y%M%H%M"`

echo "Backup old models to <$BACKUP_DIR>"

cd $MODEL_DIR
mkdir -p $BACKUP_DIR
mv *.h5 $BACKUP_DIR

echo "Download latest models from <HOST>..."

for i in ${BANKS[@]};   
do 
    echo "  downloading $i ..."
    wget http://<HOST>/model/captcha.$i.h5 
done  

echo "Restarting container $CONTAINER_ID ..."

cd $HOME
docker restart $CONTAINER_ID
