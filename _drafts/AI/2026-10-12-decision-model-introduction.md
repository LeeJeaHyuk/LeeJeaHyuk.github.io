---
layout: single
post_type: thought
title: "Decision Model (1): 작은 판단부터 바꿔보기"
date: 2026-10-12 00:00:00 +0900
last_modified_at: 2026-10-13
categories: [AI]
tags: [생각, Jev, Kev, DecisionModel, AIAgent]
permalink: /ai/decision-model-introduction/
excerpt: "Jev에 관심을 가진 계기와 공개 결정 모델의 활용 사례, 작은 판단에 적용할 때 기대한 점을 기록한다."
toc: false
---

얼마 전 OpenAI에서 연구했던 사람이 만든 Jev가 환각이 없다는 소개를 봤다. 최근에는 비슷한 방식의 결정 모델들이 Hugging Face에서 오픈소스로 공유되고, 실제로 사용하는 사례들도 올라오는 것을 보니 이제 확인해보지 않을 수 없어서 해보기로 했다. [Jev 공식 소개](https://typesafe.ai/blog/introducing-system-one-models-and-jev)  [Decision model 개념](/ai/jev-system-one-decision-model/)

사례들을 찾아보니 문서 분류나 고객 문의 처리, 브라우저에서 다음 조작을 고르는 방식으로 사용되고 있었다. 추가 학습으로 성능을 높인 사례도 있었다. 이런 사례들은 계속 업데이트되고 있으므로 앞으로도 검색해볼 사안으로 보여서, 찾아본 내용은 [자료 조사](/ai/decision-model-references/)에 정리하고 이후에도 업데이트할 예정이다.

기대한 점은 기존 LLM에서는 환각과 틀린 판단을 둘 다 고려해야 한다면, 정해진 선택지 중 하나를 고르는 구조에서는 환각을 처리하는 부담을 줄이고 틀린 판단에 집중할 수 있다라는 점이다. 또한 더 빠르고 작은 모델로 사용할 수 있다면, 작은 기업에서도 직접 학습해서 사용할 수도 있겠다 싶었다. 제한적이지만 확실한 강점이 있는 모델이라고 생각된다.

선택 기준이 명확하고 후보가 정리돼 있는 동작에서는 써볼 모델 후보로 남겨둘 수 있을 것 같다. 프로세스가 안정화되고 판단 기준과 데이터가 쌓이고 나면, 그 데이터로 훈련해서 반복적인 판단을 교체하는 식의 최적화에 사용할 수 있을지도 모른다. 그때는 훈련에 쓰지 않은 데이터에서 기존 방식과 비교하고, 틀린 판단이나 기존 LLM으로 돌아가는 비용까지 같이 봐야 할 것 같다.

게임에 적용해본 이야기는 [Decision Model (2)](/ai/minecraft-agent-lab-qwen-kev-gameplay/)에서 이어진다.
