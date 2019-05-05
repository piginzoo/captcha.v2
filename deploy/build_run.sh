version=`date +%Y%m%d%H%M`

#生成镜像文件,必须要退到上层目录，因为dockerfile add/cp都是只认context路径
cd .. 

#no cache version
docker build -f deploy/Dockerfile -t production/captcha.img:$version .
#docker build --no-cache=true -t production/captcha.img:$version .

#stop停止所有容器
docker stop $(docker ps -a -q)

#创建容器，把容器中的端口8080映射到外部的80端口上，并且mount模型目录
docker run -d --restart=always --name captcha-$version -p 80:8080 -v /home/captcha/model:/home/captcha/model -v /home/captcha/log:/home/captcha/log production/captcha.img:$version
