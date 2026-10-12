---
layout: single
post_type: reference
title: "Decision Model 자료 조사: 공개 모델과 적용 사례"
date: 2026-10-12 00:00:00 +0900
last_modified_at: 2026-10-12
categories: [AI]
tags: [자료조사, Jev, Kev, OpenJev, Mica, DecisionModel, AIAgent, Minecraft]
permalink: /ai/decision-model-references/
excerpt: "공개 결정 모델과 문서·브라우저·게임 적용 사례를 찾아보고, 원문과 평가 조건, 이후 다시 확인할 질문을 모은다."
toc: true
toc_label: "이 글의 내용"
draft: true
---

결정 모델을 직접 연결해볼 수 있는 공개 구현과 적용 사례를 찾았다. 무엇이 공개돼 있고, 어떤 역할에 사용됐으며, 품질 개선이라는 표현이 어떤 비교에서 나왔는지 정리한다. 조사일은 **2026-10-12**이며, 같은 주제의 후속 검색도 이 문서에 이어서 남긴다.

관심을 가진 계기는 [경제 프로젝트 생각 글](/ai/decision-model-economics/)과 [게임 생각 글](/ai/minecraft-agent-lab-qwen-kev-gameplay/)에서, API 설명은 [개념 정리](/ai/jev-system-one-decision-model/)에서 읽을 수 있다.

## 검색한 질문과 검색어

| 찾으려던 내용 | 사용한 검색어 | 읽은 자료 |
| --- | --- | --- |
| Jev와 공개 구현은 어떤 관계인가? | `Jev Kev decision model open source Hugging Face` | TypeSafe 소개·API 문서, Kev 저장소와 모델 카드, OpenJev·Mica 모델 카드 |
| 자기 데이터로 학습해 개선한 사례가 있는가? | `site:github.com/jaredpalmer/kev finetune accuracy CFPB` | Kev의 추가 학습 예제와 모델 카드 |
| ‘결정 모델’은 같은 답을 보장하는 결정론적 구조인가? | `Jev architecture deterministic Archer Hume` | Archer Hume의 Jev API 조사 |

검색 결과에서 원문을 따라 읽었고, 게임 사례는 공식 소개와 연결된 저장소까지 확인했다.

## 공개 모델과 제공 범위

| 모델·구현 | 확인한 제공 형태 | 다시 볼 조건과 원문 |
| --- | --- | --- |
| Jev / TypeSafe | 호스팅된 System One API | 공개 Kev 가중치와 별개다. ‘환각 없음’의 근거는 출력 스키마 보장이다. [공식 소개](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [API](https://docs.typesafe.ai/introduction) |
| Jared Palmer의 Kev | Qwen 기반 구현과 0.8B·4B·9B·27B 모델 | 자기 라벨로 추가 학습할 수 있다. 조사한 저장소 버전은 `2ecafc49`이다. [고정 버전](https://github.com/jaredpalmer/kev/tree/2ecafc49fe1e8945cd600443e6c04c1cf0b3a78f), [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) |
| OpenJev-1.7B | 로컬 결정 API, 코드·가중치·학습 및 평가 설명 | TypeSafe와 독립된 프로젝트이며 브라우저 조작·대상 선택도 다룬다. [모델 카드](https://huggingface.co/alanhuangya/OpenJev-1.7B), [코드](https://github.com/alanhuangyoo/OpenJev) |
| Mica v0.1 4B | Qwen3.5-4B 기반 BF16·GGUF 가중치와 서버 코드 | 모델 카드가 지정한 가중치는 `ca36594`이다. 언어·입력 길이·실행 정밀도에 따라 평가를 나눠 읽는다. [모델 카드](https://huggingface.co/sky7350/Mica-v0.1-4B), [코드](https://github.com/akivet/Mica-v0.1-4B) |

작은 모델이라고 모두 같은 GPU에서 실행되지는 않는다. [Kev-0.8B 카드](https://huggingface.co/jaredpalmer/kev-0.8b)의 4GB GPU 안내와 [Kev-4B 카드](https://huggingface.co/jaredpalmer/kev-4b)의 24GB GPU 예시는 다른 구성을 가리킨다. 기반 모델·adapter·출력부·캐시와 게임 실행에 필요한 메모리를 함께 봐야 한다.

## 공개 자료에서 보고한 적용 사례

| 사례 | 맡긴 역할과 보고된 결과 | 비교할 때 함께 볼 조건 |
| --- | --- | --- |
| [Kev 고객 문의 추가 학습](https://github.com/jaredpalmer/kev/blob/2ecafc49fe1e8945cd600443e6c04c1cf0b3a78f/skills/kev-finetune/README.md) | 문의에 대한 세 질문을 학습한 예제에서 정확도 67.7% → 73.6%, 오류 예산 5%일 때 자동 처리 비중 34% → 48%를 보고했다. | 생성 데이터 1,050개, H100에서 15분 학습한 특정 업무 예제이다. 데이터 400개에서는 개선이 잡음 범위 안이었다. |
| [Kev 금융 민원 데이터 학습](https://huggingface.co/jaredpalmer/kev-4b) | CFPB 민원 문서로 상품·주요 문제를 분류하는 질문을 학습했다. | 학습 기록의 실제 민원 5,219개는 투자 예측이나 수익률 실험을 뜻하지 않는다. 다른 문서로 옮겼을 때의 품질은 별도로 봐야 한다. |
| [OpenJev 브라우저 단계 평가](https://huggingface.co/alanhuangya/OpenJev-1.7B) | 다음 조작과 대상 요소를 고르는 데 사용했다. Mind2Web 평가에서 두 선택을 모두 맞힌 단계 비율 68.2%를 보고했다. | 873개 요청의 단계 평가이다. 후보 요소는 에이전트가 추렸고, 이는 브라우저 과제 전체의 완료율이 아니다. Jev 자체와는 비교하지 않았다. |
| [Mica Tetris 데모](https://huggingface.co/sky7350/Mica-v0.1-4B#demo-tetris-mica-vs-n) | 같은 조각과 선택지로 진행한 쉬운 설정에서 세 seed의 제거 줄 합계가 Mica 223, Kev-4B 55, Laya 17이었다. | 세 seed의 특정 게임 설정이다. 코드·로그·영상 생성기가 공개돼 있지만 다른 게임이나 NPC의 품질로 일반화할 수 없다. |
| [TypeSafe Doom·Wikiracing 데모](https://typesafe.ai/blog/introducing-system-one-models-and-jev#fun-demos) | 게임 행동 또는 현재 페이지의 링크를 골랐다. | Doom은 구조화된 상태를 받으며 화면 인식 데모가 아니다. Wikiracing은 많은 후보에서 점수화 후 선택하는 두 단계도 사용했다. |
| [jevman Pac-Man 비교](https://github.com/opper-ai/jevman-benchmark/tree/b202e2cefd3dcf62b99835b319450b687960f40a) | 이동 선택과 위험 회피를 결정 모델에 맡기고 게임 기록·지연·점수를 비교한다. | 응답이 제때 오지 않으면 규칙 기반 동작이 대신한다. 모델 점수와 함께 이런 대체 동작의 조건도 읽어야 한다. |

추가 학습이나 게임 데모에서 개선한 사례는 있다. 다만 같은 평가 입력과 라벨에서 비교한 것인지, 특정 분포에 맞춘 학습인지, 모델 밖 실행 코드가 얼마나 도왔는지를 함께 읽어야 내 프로젝트에서 비교할 질문을 정할 수 있다.

## API와 아키텍처를 읽을 때 필요한 자료

질문 유형은 [Primitives](https://docs.typesafe.ai/primitives), 확률에서 `confidence`를 계산하는 방법은 [Confidence](https://docs.typesafe.ai/confidence), RLCD와 보정의 취지는 [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)에 정리돼 있다.

[Archer Hume의 조사](https://archerhume.com/posts/jevs-architecture-unmasked)는 Jev API를 관찰해 내부 구조를 추론하고 반복 요청·선택지 순서의 영향을 다룬다. API에서 관찰한 현상과 제안한 아키텍처 가설을 구분해 읽는다. Kev의 공개 구조가 Jev의 비공개 구현과 같다는 근거로 옮겨 쓰지 않는다.

## 내 프로젝트의 기록과 연결

Minecraft Agent Lab은 [7be99fa 버전](https://github.com/LeeJeaHyuk/minecraft-agent-lab/tree/7be99faac450f3031798a51185bd7d36ff79c958)의 Qwen3.8-27B·Kev-4B 조합을 다룬다. 실행 조건과 야간 채집·전투·귀가 결과, 크리퍼 피해와 운영자 개입은 [Minecraft 실험 기록](/ai/minecraft-agent-lab-experiment/)에 모았다.

[경제 프로젝트 실험 기록](/ai/decision-model-economics-experiment/)은 별도 문서로 연결한다. 위 표의 금융 민원 분류 사례와는 구분한다.

## 후속 조사에서 확인할 것

- 모델을 바꿀 때 가중치 revision, 기반 모델, 언어와 컨텍스트 조건을 확인한다.
- 자기 업무의 고정된 평가 데이터에서 기존 방식과 정확도·보정·전체 작업 시간을 비교한 자료를 찾는다.
- 게임 사례는 판단 시간과 행동 실행 시간을 나누고, 규칙 기반 대체 동작과 운영자 개입을 확인한다.
- 더 작은 모델이나 양자화를 쓴 결과는 같은 입력에서 품질과 메모리 사용을 함께 비교한다.
- Jev의 첫 공개일은 날짜가 명시된 공식 자료가 있는지 다시 확인한다. 현재 소개 페이지에는 공개일이 표시되지 않는다.

## 조사 이력

| 조사일 | 추가·수정한 내용 |
| --- | --- |
| 2026-10-12 | Jev·Kev의 관계, OpenJev·Mica 공개 자료, 업무 추가 학습과 브라우저·게임 사례, API 해석에 필요한 자료를 정리했다. |
