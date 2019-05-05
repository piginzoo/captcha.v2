#-*- coding:utf-8 -*-  
#__author__ = 'piginzoo'
#__date__ = '2018/4'
from git import Repo
import argparse,re

#这个文件用于git发布，由于发布系统没办法帮我们控制tag的生成
#我们自己写一个把，规则是从version文件中读出版本模式,如r1.x.yyyyyyy
#x是我们要生成的号，
#y是用户填写一个注释，英文的
if __name__ == '__main__':
    
    repo = Repo(".")

    #解决stdout显示中文的问题
    #python3.x代码
    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser()
    parser.add_argument("--comment",help="--comment 注释(必须是英文，不能有空格）")

    args = parser.parse_args()
    comment = args.comment
    if comment == None:
        parser.print_help()
        exit()

    with open("version") as f:
        version = f.read()

    main_version = re.findall(r"^(\w+)\.",version)
    sub_version = re.findall(r"\.(\d+)\.",version)
    if len(main_version) ==0:
        print("无法找到主版本号")
        exit()
    main_version = main_version[0]   
    if len(sub_version) ==0:
        print("无法找到子版本号")
        exit()
    sub_version = sub_version[0]    

    tag = main_version + "." \
        + str(int(sub_version)+1) + "." \
        + comment
    
    with open("version","w+") as f:
        f.write(tag)

    git = repo.git()
    git.add('version') 
    git.commit('-m', 'update version file') 
    remote = repo.remote()
    remote.push()

    new_tag = repo.create_tag(tag)     
    repo.remotes.origin.push(new_tag)

    print ("新的Tag为:%s"%(new_tag))