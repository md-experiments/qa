import sys
if __name__ == '__main__':
    txt=sys.argv[1]

    print(len(txt.split()),'tokens &',len(txt),'chars')