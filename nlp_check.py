from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from collections import Counter
from konlpy.tag import Kkma
from konlpy.utils import pprint

def filler_words_check(txt):
    # filler words 리스트
    filler_words = ['설마', '그렇군요', '그렇구나', '그럼', '아야', '마구', '그러니까', '말하자면', '그다지', '어머나', '맞아요', '저', '있잖아', '아', '그래', '뭐랄까', '그', '뭐라고', '글쎄', '솔직히', '뭐지', '뭐더라', '그래요', '아무튼', '에이', '막', '아이고', '예', '어머', '세상에', '자', '뭐', '우와', '그게', '글쎄요', '정말', '음', '맞아', '어쨌든', '좀', '야', '진짜', '별로', '네', '참', '에휴', '쉿', '어', '저기요', '그냥']

    # 토큰화
    word_tokens = word_tokenize(txt)

    # filler words 체크
    result = []
    result=[word for word in word_tokens if word in filler_words]
    count = Counter(result)

    # 출력
    # print(word_tokens)
    print('사용한 fillerwords : {}'.format(result))
    print('총 사용 횟수 : {}'.format(len(result)))
    print('filler words별 사용 횟수 : {}'.format(count))

    # 시각화
    num = []
    info = []
    for i in count :
        num.append(count[i])
        info.append(i)

    colors = ['#f1bbba', '#a79c8e', '#f8ecc9']
    plt.rc('font', family='malgun gothic')
    plt.figure(figsize=(8,8))
    plt.pie(num, labels=info, autopct="%0.1f%%", colors=colors, textprops={'fontsize': 18})
    plt.title("사용한 filler words", fontsize=30)
    plt.savefig('filler_words.jpg')

def ttr_check(txt):
    # kkma 객체 생성
    kkma = Kkma()
    # 형태소 및 태그 추출
    pos = kkma.pos(txt)
    # 빈도 카운트 및 저장(dict)
    count = Counter(pos)
    # pprint(count)

    # token
    ttr_token = sum(count.values())
    # type
    ttr_type = len(count.keys())
    # TTR
    ttr = (ttr_type / ttr_token) *100

    # 출력
    # print(ttr_token, ttr_type)
    print('TTR은 : {} 입니다.'.format(ttr))

    return count

def word_end_check(count):
    word_a = 0
    word_b = 0
    for i in count.keys() :
        # 의문, 청유형 횟수 카운트
        if i[1] in ('EFQ','EFA') :
            word_a += count[i]
        # 평서, 존칭형 횟수 카운트
        elif i[1] in ('EFN','EFR') :
            word_b += count[i]

    # print(word_a)
    # print(word_b)
    # 참여유도형 화법 비율
    rate1 = word_a / (word_a+word_b) * 100
    # 공식적인 화법 비율
    rate2 = word_b / (word_a+word_b) * 100
    print('공식적인 화법(평서, 존칭형) 비율 {}  :  참여유도형 화법(의문, 청유형) 비율 {}'.format(rate2, rate1))

if __name__ == '__main__':
    f = open('주아.txt', 'r', encoding='utf-8')
    jua = f.read()
    filler_words_check(jua)
    txt_count = ttr_check(jua)
    word_end_check(txt_count)

