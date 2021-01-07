### 상황 1. fast-foward

> fast-foward는 feature 브랜치 생성된 이후 master 브랜치에 변경 사항이 없는 상황

1. feature/test branch 생성 및 이동

   ```bash
   $ git brahcn feature/test
   $ git branch
   	feature/test
   * master
   $ git checkout feature/test
   Switched to branch 'feature/test'
   (feature/test) $
   ```

   

2. 작업 완료 후 commit

   ```bash
   $ touch test.txt
   $ git add .
   $ git commit -m 'Complete test'
   [feature/test b6c6941] Complete test
    1 file changed, 0 insertions(+), 0 deletions(-)
    create mode 100644 test.txt
    $ git log --oneline
   b6c6941 (HEAD -> feature/test) Complete test
   09ff903 (master) fast-foward
   ```
   
   


3. master 이동

   ```bash
   $ git checkout master
   Switched to branch 'master'
   ```
   
   


4. master에 병합

   ```bash
   $ git merge feature/test
   Updating 09ff903..b6c6941
   # faster-forward!!!
   # MASTER
   Fast-forward
    test.txt | 0
    1 file changed, 0 insertions(+), 0 deletions(-)
    create mode 100644 test.txt
   ```
   
   


5. 결과 -> fast-foward (단순히 HEAD를 이동)

   ```bash
   $ git log --oneline
   b6c6941 (HEAD -> master, feature/test) Complete test
   09ff903 fast-foward
   ```

   

6. branch 삭제

   ```bash
   $ git branch -d feature/test
   Deleted branch feature/test (was b6c6941).
   ```
   
   

---

### 상황 2. merge commit

> 서로 다른 이력(commit)을 병합(merge)하는 과정에서 다른 파일이 수정되어 있는 상황
>
> git이 auto merging을 진행하고, commit이 발생된다.

1. feature/signout branch 생성 및 이동

   ```bash
   $ git checkout -b feature/data
   Switched to a new branch 'feature/data'
   ```

   

2. 작업 완료 후 commit

   ```bash
   $ touch data.txt
   $ git add .
   $ git commit -m 'Complete data'
   [feature/data 348141f] Complete data
    1 file changed, 0 insertions(+), 0 deletions(-)
    create mode 100644 data.txt
   ```

   

3. master 이동

   ```bash
   $ git checkout master
   Switched to branch 'master'
   ```

   

4. *master에 추가 commit 이 발생시키기!!*

   * **다른 파일을 수정 혹은 생성하세요!**

   ```bash
   $ touch hotfix. txt
   $ git add .
   
   $ git log --oneline
   6188039 (HEAD -> master) hotfix
   b6c6941 Complete test
   09ff903 fast-foward
   ```

   

5. master에 병합

   ```bash
   $ git merge feature/data
   Merge made by the 'recursive' strategy.
    data.txt | 0
    1 file changed, 0 insertions(+), 0 deletions(-)
    create mode 100644 data.txt
   ```

   

6. 결과 -> 자동으로 *merge commit 발생*

   * vim 편집기 화면이 나타납니다.

   * 자동으로 작성된 커밋 메시지를 확인하고, `esc`를 누른 후 `:wq`를 입력하여 저장 및 종료를 합니다.
      * `w` : write
      * `q` : quit
      
   * 커밋이  확인 해봅시다.

      ```bash
      $ git log --oneline
      749fae2 (HEAD -> master) Merge branch 'feature/data'
      6188039 hotfix
      348141f (feature/data) Complete data
      b6c6941 Complete test
      09ff903 fast-foward
      ```

      

7. 그래프 확인하기

   ```bash
   $ git log --oneline --graph
   *   749fae2 (HEAD -> master) Merge branch 'feature/data'
   |\
   | * 348141f (feature/data) Complete data
   * | 6188039 hotfix
   |/
   * b6c6941 Complete test
   * 09ff903 fast-foward
   ```

   

8. branch 삭제

   ```bash
   $ git branch -d feature/data
   Deleted branch feature/data (was 348141f).
   ```
   
   ```bash
   $ git log --oneline --graph
   *   749fae2 (HEAD -> master) Merge branch 'feature/data'
   |\
   | * 348141f Complete data # 브랜치(feature/data)가 사라짐
   * | 6188039 hotfix
   |/
   * b6c6941 Complete test
   * 09ff903 fast-foward
   ```
   
   

---

### 상황 3. merge commit 충돌

> 서로 다른 이력(commit)을 병합(merge)하는 과정에서 동일 파일이 수정되어 있는 상황
>
> git이 auto merging을 하지 못하고, 해당 파일의 위치에 라벨링을 해준다.
>
> 원하는 형태의 코드로 직접 수정을 하고 merge commit을 발생 시켜야 한다.

1. feature/board branch 생성 및 이동

   ```ㅠㅁ노
   $ git checkout -b feature/web
   Switched to a new branch 'feature/web'
   ```

   

2. 작업 완료 후 commit

   


3. master 이동

   


4. *master에 추가 commit 이 발생시키기!!*

   * **동일 파일을 수정 혹은 생성하세요!**
   

   
5. master에 병합

   ```bash
   $ git merge feature/web
   # 자동으로 병합하는 중에..
   Auto-merging readme.md
   # 충돌발생(Merge conflict)
   CONFLICT (content): Merge conflict in readme.md
   # 자동 머지 실패함;
   # 충돌을 고치고 결과를 커밋해.
   Automatic merge failed; fix conflicts and then commit the result.
   (master|MERGING) $
   ```
   
   


6. 결과 -> *merge conflict발생*

   ```bash
   $ git status
   On branch master
   You have unmerged paths.
     (fix conflicts and run "git commit")
     (use "git merge --abort" to abort the merge)
   
   Changes to be committed:
           new file:   web.txt
   # 어디에 충돌이 난건지 확인..
   Unmerged paths:
     (use "git add <file>..." to mark resolution)
           both modified:   readme.md
   ```
   
   


7. 충돌 확인 및 해결

   ```bash
   <<<<<< HEAD
   # Project
   
   * data 프로젝트
   =======
   # 프로젝트
   
   * web개발
   >>>>>>> feature/web
   ```
   
   


8. merge commit 진행

    ```bash
    $ git commit
    ```

   * vim 편집기 화면이 나타납니다.
   
   * 자동으로 작성된 커밋 메시지를 확인하고, `esc`를 누른 후 `:wq`를 입력하여 저장 및 종료를 합니다.
      * `w` : write
      * `q` : quit
      
   * 커밋이  확인 해봅시다.
   
9. 그래프 확인하기

    ```bash
   $ git log --oneline --graph
   *   302c8c6 (HEAD -> master) Merge branch 'feature/web'
   |\
   | * 5dd5b0f (feature/web) Update README and Complete web
   * | 566baa8 Update README
   |/
   *   749fae2 Merge branch 'feature/data'
   |\
   | * 348141f Complete data
   * | 6188039 hotfix
   |/
   * b6c6941 Complete test
   * 09ff903 fast-foward
    ```
   
   


10. branch 삭제

    
