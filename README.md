# 이재혁의 엔지니어링 기록

`https://leejeahyuk.github.io`에서 제공하는 Jekyll 기반 기술 블로그입니다.

로컬 LLM, AI 에이전트, 데이터 파이프라인을 실제로 운영하며 발견한 문제와 선택, 결과를 기록합니다.

## 로컬 실행

```bash
bundle install
bundle exec jekyll serve --livereload
```

초안을 포함해 확인하려면 다음과 같이 실행합니다.

```bash
bundle exec jekyll serve --drafts --livereload
```

## 글 작성 원칙

각 글은 하나의 큰 문제와 하나의 결정을 중심으로 작성합니다.

1. 해결하려던 문제
2. 발견한 실패 또는 제약
3. 검토한 대안
4. 선택과 이유
5. 측정된 결과
6. 남은 불확실성과 재검토 조건
