---
layout: single
post_type: thought
title: "Decision Model (1): 작은 판단부터 바꿔보기"
date: 2026-10-12 00:00:00 +0900
last_modified_at: 2026-10-12
categories: [AI]
tags: [생각, Jev, Kev, DecisionModel, AIAgent]
permalink: /ai/decision-model-economics/
excerpt: "Jev에 관심을 가진 계기와 공개 결정 모델의 활용 사례를 따라가며, 경제 프로젝트의 작은 판단부터 바꿔보기로 한 과정을 기록한다."
toc: false
---

얼마 전 OpenAI에서 연구했던 사람이 만든 Jev가 환각이 없다는 소개를 봤다. 최근에는 비슷한 방식의 결정 모델들이 Hugging Face에서 오픈소스로 공유되고, 실제로 사용하는 사례들도 올라오는 것을 보니 이제 확인해보지 않을 수 없어서 해보기로 했다. [Jev 공식 소개](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

기술 설명만 읽고 싶다면 [개념 정리](/ai/jev-system-one-decision-model/)로, 경제 프로젝트에서 적용한 내용은 [실험 기록](/ai/decision-model-economics-experiment/)으로 바로 갈 수 있다.

사례들을 찾아보니 문서 분류나 고객 문의 처리, 브라우저에서 다음 조작을 고르는 방식으로 사용되고 있었다. 추가 학습으로 성능을 높인 사례도 있었다. 이런 사례들은 계속 업데이트되고 있으므로 앞으로도 검색해볼 사안으로 보여서, 찾아본 내용은 [자료 조사](/ai/decision-model-references/)에 정리하고 이후에도 업데이트할 예정이다.

기대한 점은 기존 LLM에서는 환각과 틀린 판단을 둘 다 고려해야 한다면, 정해진 선택지 중 하나를 고르는 구조에서는 이상한 출력을 처리하는 부담을 줄이고 틀린 판단에 집중할 수 있겠다는 점이었다. 더 빠르고 작은 모델로 사용할 수 있다면, 데이터와 판단 기준이 풍부한 기업에서는 직접 학습해서 사용할 수도 있겠다 싶었다. 제한적이지만 확실한 강점이 있는 모델이라고 생각했다.

그래서 바로 실제 적용한 사례를 공부하고 공개된 오픈소스를 사용해서 개인 프로젝트에 해보았다. 경제 프로젝트에서는 먼저 기존 프로세스에서 판단이 필요한 부분을 각각 선별하고, 각각을 결정 모델 방식으로 변경해본 뒤 기존 방식과 비교해보았다.

<!-- 경제 프로젝트 담당 세션에서 보완: 프로젝트 이름·링크, 교체한 각 판단과 선택지, 비교 방법, 관찰 결과, 그 결과로 바뀐 생각. 구체적인 성과나 수치는 현재 넣지 않는다. 실험 기록의 대응 항목을 먼저 채운 뒤 이 문단을 이어 쓴다. -->

그다음에 게임에 적용해본 이야기는 [Decision Model (2)](/ai/minecraft-agent-lab-qwen-kev-gameplay/)에서 이어진다.
