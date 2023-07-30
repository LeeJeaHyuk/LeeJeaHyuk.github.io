---
layout: single
title: "[Analytics on AWS] - Catalog Data"
categories: [aws], [Analytics on AWS]
tag : [aws], [Analytics on AWS]
toc : true
---

![Architecture Diagram](https://static.us-east-1.prod.workshops.aws/public/9b2d1982-fdcf-4207-ba26-71a458796115/static/images/catalog.png?classes=shadow)

- 학습 목표

  - AWS Glue 데이터 카탈로그에 데이터셋을 등록

  - Glue 크롤러를 이용하여 메타데이터 수집을 자동화

  - 카탈로그 엔티티가 생성되면, Amazon Athena를 사용하여 데이터의 원시 형식을 쿼리할 수 있게 된다

    - AWS Glue는 AWS에서 제공하는 완전 관리형 ETL(Extract, Transform, Load) 서비스 : 데이터의 이동, 변환, 정제 등의 작업을 자동화

    - ETL은 데이터를 추출하여 변환한 뒤 원하는 형식으로 로드하는 작업

---

## IAM Role 생성

IAM 콘솔 https://us-east-1.console.aws.amazon.com/iamv2/home#/roles로 이동하여 새 AWS Glue service role을 생성합시다.