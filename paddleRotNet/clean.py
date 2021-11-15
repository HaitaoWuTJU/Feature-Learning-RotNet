# -*-coding:utf-8-*- 
import os
SAVELIST=['main.py']
def deleteNouese(path):#遍历指定文件夹中所有文件，检查图像大小，长高小于300的删除,不是图像的文件也删除
    for root,dirs,files in os.walk(path):
        for file in files:
            if file in SAVELIST:
                continue
            name=file.rsplit('.',1)[0]
            postfix=file.rsplit('.',1)[-1]
            # print(file.rsplit('.',1))
            if postfix=='py':
                pycPath=os.path.join(root,'__pycache__',name+'.cpython-38.pyc')
                flag=os.path.exists(pycPath)
                # print(name+'.cpython-38.pyc')
                if flag:
                    print('save ',os.path.join(root,'__pycache__',name+'.cpython-38.pyc'))
                    pass
                else:
                    pypath=os.path.join(root,file)
                    print('delete {}'.format(pypath))
                    print('delete {}'.format(pycPath))
                    # print(file+' '+(name+'.cpython-38.pyc'))
            # print(os.path.join(root,name))
            # aa1=os.path.join(root,name)
            # if aa1.split('.')[-1]=='pyc':
            #     print('del')
            #     os.remove(aa1)#删除文件

def deletepyc(path):
    for root,dirs,files in os.walk(path):
        for file in files:
            aa1=os.path.join(root,file)
            # print(aa1)
            if aa1.split('.')[-1]=='pyc':
                print(aa1)
                os.remove(aa1)#删除文件
def main():
    path="./"
    deletepyc(path)
    # ab(path)
if __name__=="__main__":
    main()