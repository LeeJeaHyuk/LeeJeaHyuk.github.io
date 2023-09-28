---
layout: single
title: "Obsidian Plugin(1) Dataview and Tasks"
categories: [Obsidian]
tags:
  - Post
  - Obsidian
  - Dataview
  - Tasks
toc : true
---


이번 포스트에서는 옵시디언을 사용하면서 잘 사용하고 있는 플러그인들 중 Dataview를 소개하려고 한다.

마크다운 파일들을 생성하면서 지정해놓은 Tag, Property를 사용하여 저장해놓은 파일들을 한눈에 볼 수 있게 도와준다.

Dataview는 크게 3종류의 시각적 표현 방식을 제공하는데 이는 "Table," "List," "Task"이다.

---

## Table
먼저 Table부터 보면, 나는 취업을 준비중이기 때문에 다양한 회사들이 어떤 요구사항과 우대사항을 공고에 쓰는지 한 눈에 알 수 있는 표를 만들려고 했다.

먼저 표에 들어갈 데이터들을 직접 공고 사이트에서 입력하였다.


![스크린샷 2023-09-28 142159](../../images/Obsidian Plugin(1) Dataview and Tasks/스크린샷 2023-09-28 142159.png)

파일의 내용에는 일단 모든 요구사항과 우대사항 업무 등을 전부 적어주고 Tag와 Property 또한 내가 분류하려는 만큼 적당히 지정해준다

![image-20230928142636033](../../images/Obsidian Plugin(1) Dataview and Tasks/image-20230928142636033.png)

또 파일에서 내가 한눈에 보고싶은 내용들을 "::" 를 사용하여 미리 지정해놓는다.

![image-20230928142903108](../../images/Obsidian Plugin(1) Dataview and Tasks/image-20230928142903108.png)

나는 내가 준비할 요구사항들을 한눈에 보고 싶었기 때문에  Rrquirements:: 로 지정해주었다.



이후 Dataview를 사용하여 표를 만들어 주면 

![제목 없음](../../images/Obsidian Plugin(1) Dataview and Tasks/제목 없음.png)
신입 태그의 모든 정보를 가져와서 만든 날짜와 아까 지정한 내가 준비하려고 하는 요구사항들을 한눈에 볼 수 있게 된다. 또한 File열에서 각각의 파일과 연결되어있기 때문에 추가적인 정보를 확인하고 싶으면 바로 링크를 따라서 들어갈 수 있다.

---

## list

Dataview에서 list는 특정 Tag들을 가져와서 한눈에 볼 수 있게 해 준다.
Obsidian에서는 마인드맵을 작성할 때 처럼 계속해서 꼬리를 물고 파일들이 이어질 수 있는데 각각의 파일을 한번에 완성시키지 못할 때가 있다. 이 경우에 나는 Incomplete 라는 Tag를 지정하여 한번에 불완전한 파일들을 몰아볼 수 있게 된다.

![list](../../images/Obsidian Plugin(1) Dataview and Tasks/list.png)

---

## Task

마지막으로 Task가 있는데 task는 만들어 놓은 tasks들을 한눈에 보기 쉽게 만들어준다.

먼저 Tasks 플러그인에 대해 알아보자 Tasks Plugin은 cmd(Windows : ctrl) + P 를 사용해서 Create or edit Task를 선택하면 아래와 같은 창이 나온다.

![image-20230928214541456](../../images/Obsidian Plugin(1) Dataview and Tasks/image-20230928214541456.png)

이 창에서 할일과 중요도 매일 반복하는 일인지, 마감기한 등등을 지정해주고 만들 수 있다. 

나는 매일 해야 할 일을 지정하려고 생각했기 때문에 데잍리노트를 활용했다.
데일리노트는 클릭하면 아래와 같이 오늘 날짜로 된 마크다운 파일을 생성한다. 
![image-20230928215218577](../../images/Obsidian Plugin(1) Dataview and Tasks/image-20230928215218577.png)

원래는 아무것도 없는 비어있는 파일이 생성되지만 설정에 데일리 노트를 확인하면 미리 Template을 설정해놓을 수 있다 한 번 설정해놓으면 데일리노트를 만들 때마다 Template에 지정한 파일이 이름만 바뀌어서 대신 나오게 되고, 데일리 노트를 모아 놓을 경로 또한 미리 지정해놓을 수 있다.

나는 Eisenhower Decision 을 사용해서 할 일의 우선순위를 정하기 위해 템플릿에 적어놓았다.

![image-20230928215416992](../../images/Obsidian Plugin(1) Dataview and Tasks/image-20230928215416992.png)

이와 같이 각각의 날에 Tasks를 정하고 나서 모든 tasks를 한 눈에 볼 수 있도록 해주는 것이 dataview의 task이다.

아래와 같이 사용한다면 전체 task를 가져와서 완료되지 않은 task들을 각각의 file link에 맞게 분류하여 볼 수 있다.


![image-20230928214125471](../../images/Obsidian Plugin(1) Dataview and Tasks/image-20230928214125471.png)

---

## 마치며

이번 포스트에서는 Obsidian의 Dataview를 사용해서 내가 저장한 파일들의 정보들 중 필요한 것을 Table을 통해 시각화하는 것과 List를 통해 완성되지 않은 파일들 같이 필요한 태그들을 모아서 보는 것, 또 tasks 플러그인과 Dataview의 task를 통해 할 일을 정리하고 무엇을 해야 할 지 시각화하여 보는 것을 확인해보았다.

다음 포스트에는 저장과 이미지 경로를 지정하는 플러그인을 소개해보려고 한다.

Next Post : []()

---

## reference
[Obsidian Dataview Github](https://github.com/blacksmithgu/obsidian-dataview)

[Obsidian Dataview Overview](https://blacksmithgu.github.io/obsidian-dataview/)

