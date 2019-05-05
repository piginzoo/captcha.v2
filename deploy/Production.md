#环境说明

##概要
此文档说明，如何配置安装环境，
环境包括三台机器：

操作系统都是Ubuntu 16.04 64位

##Web API服务器

###1. 用root身份安装软件、设置环境变量
apt-get update
apt-get install -y git
apt-get install -y lrzsz
echo "export LC_ALL=\"en_US.UTF-8\"">> /etc/rc.local
echo "export LC_ALL=\"en_US.UTF-8\"">>/home/captcha/.bashrc

###2. 创建captcha用户
useradd captcha -m
cp -r /root/.pip /home/captcha/
chown -R captcha /home/captcha/.pip

###3. 切换到captcha用户，git clone代码，创建必要目录
su captcha
cd /home/captcha
**clone from git repo**
mkdir /home/captcha/model
mkdir /home/captcha/log

###4.安装docker，使用阿里云的apt源
阿里云安装：

####step 1: 安装必要的一些系统工具
sudo apt-get update
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common

####step 2: 安装GPG证书
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -

####step 3: 写入软件源信息
sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

####step 4: 更新并安装Docker-CE
sudo apt-get -y update
sudo apt-get -y install docker-ce

###5.创建docker镜像和容器
cd /home/captcha/captcha/deploy
./build_run.sh

###6.测试服务是否正常
访问http://IP，测试web网页及页面上的验证服务