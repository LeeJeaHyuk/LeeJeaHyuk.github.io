

# sequence-to-sequence 

[seq2seq 설명 블로그](https://wikidocs.net/24996)

![image-20220712151014650](../../images/seq2seq/image-20220712151014650.png)

인코더와 디코더로 두 개의 모듈로 구성되어 있다.

1. 인코더는 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤에 마지막에 이 모든 단어 정보들을 압축해서 하나의 벡터(context vector)로 만든다.
2. 입력 문장의 정보가 하나의 context vector로 모두 압축되면 context vector를 디코더로 전송한다.
3. 디코더는 context vector를 받아서 번역된 단어를 한 개씩 순차적으로 출력한다.



context vector 예시 사진 : 실제로는 수백개의 차원을 갖고 있는 경우가 많다.

![image-20220712151306762](../../images/seq2seq/image-20220712151306762.png)





![image-20220712151344737](../../images/seq2seq/image-20220712151344737.png)

인코더와 디코더의 Architecture 내부를 살펴보면 두 개의 RNN Architecture로 구성되어있음을 볼 수 있다.

1. 인코더 : 입력 문장을 받는 RNN셀
2. 디코더 : 출력 문장을 출력하는  RNN셀 
3. 바닐라 RNN이 아니라 LSTM 셀 또는 GRU **셀**들로 구성되어 있다.



## Incoder

1. 입력 문장은 단어 코큰화를 통해서 단어 단위로 쪼개지고 단어 토큰은 각각 RNN셀의 각 시점에 입력이 된다.
   1. ![image-20220712151723818](../../images/seq2seq/image-20220712151723818.png)
2. 인코더 RNN셀은 모든 단어를 입력받은 뒤에 인코더 RNN 셀의 마지막 시점의 은틱 상태를 디코더 RNN셀로 넘겨준다.
3. 넘겨준 벡터를 context vector 라고 한다.
4. context vector는 디코더 RNN셀의 첫 번째 은닉 상태에 사용된다.



## Decoder RNNLM(RNN Language Model)

![image-20220712152640434](../../images/seq2seq/image-20220712152640434.png)

테스트 과정 동안에

1. 디코더는 초기 입력으로 문장의 시작을 의미하는 심볼 $<sos>$가 들어 있다.
   1. $<sos> == <s> == <Go> == <bos>$ 심볼은 여러 종류로 쓰이는 듯 하다.
2. 디코더는 $<sos>$가 입력되면, 다음에 등장할 확률이 높은 단어를 예측한다.
   1. 첫 번째 timestep의 디코더 RNN셀은 다음 단어를 $je$로 예측한 모습
3. 문장의 끝을 의미하는 심볼인 $<eos>$가 다음 단어로 예측될 때 까지 반복된다.
   1. $<eos> == </s> == <end> $



훈련 과정 동안에는![image-20220712153115820](../../images/seq2seq/image-20220712153115820.png)

디코더에게 인코더가 보낸 context vector와 실제 정답인 $<sos> je suis étudiant $를 입력 받았을 때

 $je suis étudiant <eos> $가 나와야 한다고 정답을 알려주면서 훈련한다.

1. seq2seq에서 사용되는 모든 단어들은 임베딩 벡터로 변환 후 입력으로 사용한다.
   1. ![image-20220712153600982](../../images/seq2seq/image-20220712153600982.png)
   2. 임베딩 층을 거쳐서 입력된다.

![image-20220712153705484](../../images/seq2seq/image-20220712153705484.png)

Word Embedding 예시 실제로 임베딩 벡터는 수백개의 차원을 가질 수 있다.



## RNN셀에서의 timestep

![image-20220712154416467](../../images/seq2seq/image-20220712154416467.png)

1. 현재 timestep을 t라고 할 때,  RNN셀은 t-1에서의 은닉 상태와 t에서의 입력 벡터를 입력으로 받는다.
2. t에서의 은닉 상태를 만든다.
3. t에서의 은닉 상태는 바로 위에 또 다른 은닉층이나 출력층이 존재할 경우에는 위의 층으로 보내거나 필요 없다면 값을 무시할 수 있다.
4. RNN셀은 다음 타임 스텝에 해당하는 t+1의 RNN셀의 입력으로 현재 t에서의 은닉 상태를 입력으로 보낸다.



## Context vector

인코더 RNN셀의 마지막 시점의 은닉 상태를 디코더로 넘겨준 벡터

