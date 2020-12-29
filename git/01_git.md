# git 기초

> 분산버전관리시스템(DVCS)

## 0. 로컬 저장소(repository) 설정

```bash
$ git init
# 초기화되었다..
Initialized empty Git repository in 
C:/Users/안종배/OneDrive/바탕 화면/practice/.git/
(master) $
```

* `.git 폴더가 생성되고, 여기에 모든 git과 관련된 정보들이 저장된다.

## 기본 작업 흐름

> 모든 작업은 touch 로 파일을 만드는 것으로 대체

### 1. add

```bash
$ git add. # . : 현재 디렉토리(하위 디렉토리 포함)
$ git add a. txt # 특정 파일
$ git add my_folder/ # 특정 폴더
$ git add a.xt b.txt c.txt #복수의 파일
```

* working directory의 변경사항(첫번째 통)을 staging area(두번째 통)상태로 변경 시킨다.
* 커밋의 대상 파일을 관리한다.

```bash
$ touch a.txt
$ git status
On branch master

No commits yet
# 트래킹이 되고 있지 않는 파일들...
# => 새로 생성된 파일
Untracked files:
	#add 명령을 사용해!
	# 커밋이 될 것에 포함시키기 위하여...
	# => Staging Area (두번째 통)
  (use "git add <file>..." to include in what will be committed)
        a.txt

nothing added to commit but untracked files present (use "git add" to track)
```

* add 이후

```bash
$ git add .
$ git status
On branch master

No commits yet
# 커밋이 될 변경사항
# SA 두번째통에 있는 애들
Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   a.txt
```

### 2. commit

```bash
$ git commit -m 'First commit'
[master (root-commit) 72be2a3] First commit
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 a.txt
```

* `commit` 은 지금 상태를 스냅샷을 찍는다.
* 커밋 메시지는 지금 기록하는 이력을 충분히 잘 나타낼 수 있도록 작성한다.
* ` git log`  명령어를 통해 지금까지 기록된 커밋들을 확인할 수 있다.

## 기타 명령어

### 1. status

> 로컬 저장소의 상태

```bash
$ git status
```

### 2. log

>  커밋 히스토리

```bash
$ git log
commit 72be2a35699141886a075b5394ae49bf0230ffce (HEAD -> master)
Author: an4509 <an4509@hanmail.net>
Date:   Tue Dec 29 14:10:55 2020 +0900

    First commit

안종배@DESKTOP-J1LBO1U MINGW64 ~/OneDrive/바탕 화면/practice (master)
$ git log --oneline
72be2a3 (HEAD -> master) First commit
$ git log -2
$ git log --oneline -1
```

## git commit author 설정

> 최초에 컴퓨터에서 git을 활용하려고 하면, 아래의 설정을 하지 않으면 commit이 안된다.

```bash
$ git config --global user.name __username__
$ git config --global user.email __email__
```

* 설정을 확일할 때는 아래의 명령어를 활용한다.

```bash
$git config --global -l
user.name=an4509
user.email=an4509@hanmail.net
```

* 설정된 이메일이 Github에 등록된 이메일이랑 같도록 하는 것을 추천(잔디밭)

# 원격저장소(remote repository) 활용 기초

> 다양한 원격저장소 서비스 중에 Github을  기준으로 설명

## 준비사항

* Github에 비어있는 저장소(repository)를 만든다.

## 기초 명령어

### 1. 원격저장소 설정

```bash
$ git remote add origin __url__
```

* 깃, 원격저장소를(remote)추가해줘(add). origin이라는 이름으로 URL!

* 설정된 원격저장소를 확인하기 위해서는 아래의 명령어를 입력한다.

```bash
$ git remote -v
origin  https://github.com/an4509/practice.git (fetch)
origin  https://github.com/an4509/practice.git (push)
```

### 2. push

```bash
$ git push origin master
```

* origin 원격저장소의 master 브랜치로 push