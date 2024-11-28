这篇文章是由VNCTF的"crypto_sign_in_5"一题有感而发，在简单学习了一下Dual attack和primal attack的思路之后，觉得确实很有记录的价值，决定新开一个"crypto-learning"类别，来记录一下学习密码的过程中碰到的各种比较有意思的知识点和技巧，就有点类似于学习笔记吧。

这些文章里提及的概念大多并不会特别详细、准确，因为更详细、准确的概念可以在网上很多博客、参考文献等找到，因此文章里面只会挑我觉得重要一些的来记录。

本文的具体实例就都围绕"crypto_sign_in_5"这一个题目。

<!--more-->

### LWE

这一部分是总体描述一下LWE，详细细节可以参考：

[初探全同态加密之二：格密码学与LWE问题 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/150920501)

#### 概念

一个LWE的样本可以描述为：
$$
A_{m\times n}s_{n\times 1} + e_{m\times 1} = b_{m\times 1} \quad(mod\;p)
$$
其中：

+ 样本公钥：A、b
+ 样本私钥：s
+ 噪声向量：e，在模p下要显得数量级较小

#### 生成过程

先生成LWE的私钥s，然后对每一组样本，随机生成一个模q下的mxn维矩阵A，以及长为m的噪声向量e，并计算得到向量b，之后给出A、b。

示例(生成一组m=128，n=77的LWE样本)：

```python
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
from sage.crypto.lwe import LWE, samples

p = 0xfffffffffffffffffffffffffffffffeffffffffffffffff
m,n = 32*4,77
esz = 2**64
F = GF(p)
V = VectorSpace(F,n)
D = DiscreteGaussianDistributionIntegerSampler(esz)

#gen private_key s
lwe = LWE(n=n, q=p, D=D)
s = lwe._LWE__s

#gen sample
sample = list(zip(*samples(m=m, n=n, lwe=lwe)))
A,b = sample
```

#### 困难问题

一般来说基于LWE的困难问题分为搜索LWE(Search-LWE)和决策LWE(Decision-LWE)两种，这里简单说明一下。

**Search-LWE(SLWE)**：已知样本是一组LWE样本，给出样本的A、b，找出样本私钥s。

**Decision-LWE(DLWE)**：给出当前样本的A、b，区分当前样本是否是LWE样本。用"crypto_sign_in_5"的代码来表示就是要区分以下两种样本：

```python
sample_LWE = list(zip(*samples(m=m, n=n, lwe=lwe)))
sample_random = list(zip(*[[V.random_element(),(F.random_element())] for _ in range(m)]))
```

"crypto_sign_in_5"就是一个DLWE问题，他给了若干组m=32，n=77的样本，每组是由flag的对应比特来决定是LWE样本还是纯随机样本，如果能成功区分出来的话就可以确定flag的各bit。

<br/>

<br/>

### 格攻击

对于LWE问题主要有primal attack和Dual attack两种攻击方式。接下来就围绕题目展开对这两种攻击方式的介绍，由于我自己只是大概学了一下，一点也不精，所以如有错误欢迎各位师傅指出。

两种攻击都有最基础的版本，以及一些优化后的版本，这里我只会介绍一些我能明白思路的版本。

#### primal attack(最基础版本)

在我的理解里，primal attack更偏向于解决SLWE问题。具体来说，由于LWE样本有：
$$
A_{m\times n}s_{n\times 1} + e_{m\times 1} = b_{m\times 1} \quad(mod\;p)
$$
思路的落脚点肯定在于e数量级较小，所以可以构造格，让e出现在目标向量中来缩短目标向量的数量级。这种造格思路就和很多利用线性等式造格的思路并没有区别，这里就直接给出对应的格：
$$
\left(\begin{matrix}
1&&&&a_{1,1}&a_{2,1}&&a_{m,1}\\
&1&&&\vdots&\vdots&\ddots\\
&&\cdots&&a_{1,n}&a_{2,n}&&a_{m,n}\\
&&&1&-b_{1}&-b_{2}&\cdots&-b_{m}\\
&&&&p\\
&&&&&p\\
&&&&&&\cdots\\
&&&&&&&p\\
\end{matrix}\right)
_{m+n+1,m+n+1}
$$
这个格具有的线性关系是：
$$
(s_1,s_2,...s_n,1,k_1,k_2,...k_{m})
\left(\begin{matrix}
1&&&&a_{1,1}&a_{2,1}&&a_{m,1}\\
&1&&&\vdots&\vdots&\ddots\\
&&\cdots&&a_{1,n}&a_{2,n}&&a_{m,n}\\
&&&1&-b_{1}&-b_{2}&\cdots&-b_{m}\\
&&&&p\\
&&&&&p\\
&&&&&&\cdots\\
&&&&&&&p\\
\end{matrix}\right)
=
(s_1,s_2,...s_n,1,e_1,e_2,...,e_{m})
$$
由于使目标向量中的值数量级相近会更利于规约，所以要给格乘对角矩阵来配平系数，这是格的一些基本操作就不再详述。

这个格中的最后m列是用来规约出e的约束，所以m越大，约束列越多，对于规约就越有帮助。因此经测试可以发现，对于"crypto_sign_in_5"这个题目，如果仅仅取一组LWE样本来找出私钥s，也就是m=32，n=77，e在64比特数量级，p在192比特数量级的情况下，是远远不够的。

但是注意到，对于私钥s相同的LWE样本，可以取多组A、b加入到格中对e的约束列里。这一点很好理解，比如为对于两组A维数为mxn的LWE样本：
$$
A_{1}s + e_{1} = b_{1} \quad(mod\;p)
$$

$$
A_{2}s + e_{2} = b_{2} \quad(mod\;p)
$$

这完全可以看作是一组A维数是2mxn的样本：
$$
(\frac{A_1}{A_2})s + (\frac{e_1}{e_2}) = (\frac{b_1}{b_2}) \quad(mod\;p)
$$
也就是把矩阵A、向量e和向量b都上下拼接起来即可。

那么对于这个题目，经测试可以发现，对于m=32，n=77，e在64比特数量级，p在192比特数量级的LWE样本，如果取四组的话(也就相当于一个m=128，n=77的LWE样本)，就可以用这个格规约出来私钥和误差向量了。

这一部分代码如下(实际操作中会发现上面的格换一换行顺序会规约的更快)：

```python
def primal_attack1(A,b,m,n,p,esz):
    L = block_matrix(
        [
            [matrix.identity(m)*p,matrix.zero(m, n+1)],
            [(matrix(A).T).stack(-vector(b)).change_ring(ZZ),matrix.identity(n+1)],
        ]
    )
    #print(L.dimensions())
    Q = diagonal_matrix([p//esz]*m + [1]*n + [p])
    L *= Q
    L = L.LLL()
    L /= Q
    for res in L:
        if(res[-1] == 1):
            s = vector(GF(p), res[-n-1:-1])
            return s
        elif(res[-1] == -1):
            s = -vector(GF(p), res[-n-1:-1])
            return s
```

但是，此时的格是206行206列的方阵，维数比较大，耗时也就比较久，用sage10.2跑需要六分钟以上，只能说勉强可以接受。

这就引出了下面要介绍的一种优化版本的primal attack。

#### primal attack(优化版本)

注意到，刚才构造的格，他规约的目标向量中有n维的s向量，这就带来两个问题：

+ s向量本身是在模p下随机取值的，他并不是个短向量
+ 他让造的格多了n维的单位阵，增大格的维度也就代表着增长耗时

而这个优化就可以避免这两个问题，他的实现思路如下。

首先仍然把LWE的等式写出来：
$$
As + e = b
$$
将式子进行转置：
$$
s^TA^T + e^T = b^T
$$
接下来就是这个优化的神来之笔，我们知道，对于任意一个nxm维的矩阵(n<m)，都可以通过初等行变换来变成行最简形矩阵，而这个行最简形矩阵可以看作是一个nxn的单位阵，在右边拼接上一个nx(m-n)的矩阵的形式。那么现在，A^T就是一个nxm的矩阵，我们假设他的行最简形矩阵是K，那么就有变换：
$$
A^T_{n\times m} \rightarrow K
$$
而由于这种变化均是初等行变换，所以可以写成是一个可逆矩阵P左乘的形式：
$$
PA^T_{n\times m} = K
$$
又因为刚才说的，这个行最简形矩阵可以看作是一个nxn的单位阵，在右边拼接上一个nx(m-n)的矩阵。所以也就有：
$$
PA^T_{n\times m} = [I_{n\times n},A'_{n \times (m-n)}]
$$
那么对于刚才的式子，我们把这样变换后的式子代进去：
$$
s^T(P^{-1}P)A^T + e^T = b^T
$$

$$
\rightarrow (s^TP^{-1})(PA^T) + e^T = b^T
$$

再令：
$$
s' = s^TP^{-1}
$$
那么最终就有：
$$
s'[I,A'] - b^T = e^T \quad(mod\;p)
$$
那么现在就用这个等式来造格，造格的思路依然和其他线性等式造格的没什么不同，造出来的格是：
$$
\left(\begin{matrix}
1&&&a'_{1,1}&a'_{2,1}&&a'_{m-n,1}&\\
&\cdots&&\vdots&\vdots&\ddots&&\\
&&1&a'_{1,n}&a'_{2,n}&&a'_{m-n,n}&\\
p\\
&p\\
&&p\\
&&&\ddots\\
&&&&\ddots\\
&&&&&\ddots\\
&&&&&&p\\
-b_{1}&-b_{2}&-b_{3}&\cdots&\cdots&\cdots&-b_{m}&1\\
\end{matrix}\right)
$$
这个格的线性关系是：
$$
(s'_1,s'_2,...,s'_n,k_1,k_2,...k_m,1)
\left(\begin{matrix}
1&&&a'_{1,1}&a'_{2,1}&&a'_{m-n,1}&\\
&\cdots&&\vdots&\vdots&\ddots&&\\
&&1&a'_{1,n}&a'_{2,n}&&a'_{m-n,n}&\\
p\\
&p\\
&&p\\
&&&\ddots\\
&&&&\ddots\\
&&&&&\ddots\\
&&&&&&p\\
-b_{1}&-b_{2}&-b_{3}&\cdots&\cdots&\cdots&-b_{m}&1\\
\end{matrix}\right)
=
(e_1,e_2,...,e_m,1)
$$
不过这样会感觉很怪，这样虽然把目标向量中的s去掉了，但这个格并不是一个方阵，行数依然是m+n+1，并没什么实质上的提升。

但最妙的一步也就在这里，注意到格的最左上角是单位阵，而s'向量是模p下的，他与单位阵计算乘积得到的每个值肯定也都小于p。因此，对于格的前n列来说，由于本身计算值一定小于p，因此是没有+kp的必要的！所以我们就可以把这个矩阵去除掉n维的冗余，得到一个方阵：
$$
\left(\begin{matrix}
1&&&a'_{1,1}&a'_{2,1}&&a'_{m-n,1}&\\
&\cdots&&\vdots&\vdots&\ddots&&\\
&&1&a'_{1,n}&a'_{2,n}&&a'_{m-n,n}&\\

&&&p\\
&&&&p\\
&&&&&\ddots\\
&&&&&&p\\
-b_{1}&-b_{2}&-b_{3}&\cdots&\cdots&\cdots&-b_{m}&1\\
\end{matrix}\right)
$$
这个格依然具有如下线性关系：
$$
(s'_1,s'_2,...,s'_n,k_{m-n+1},k_{m-n+2},...k_m,1)
\left(\begin{matrix}
1&&&a'_{1,1}&a'_{2,1}&&a'_{m-n,1}&\\
&\cdots&&\vdots&\vdots&\ddots&&\\
&&1&a'_{1,n}&a'_{2,n}&&a'_{m-n,n}&\\

&&&p\\
&&&&p\\
&&&&&\ddots\\
&&&&&&p\\
-b_{1}&-b_{2}&-b_{3}&\cdots&\cdots&\cdots&-b_{m}&1\\
\end{matrix}\right)
=
(e_1,e_2,...,e_m,1)
$$
至此，我们得到了一个完美符合要求的m+1维的方阵，通过减小n维的冗余，大大减小了LLL的时间，仅需要一分钟左右就能规约出噪声向量e了。

规约出e后，直接解如下矩阵方程：
$$
As = b-e \quad(mod\;p)
$$
就可以得到私钥s，这就解决了SLWE问题。这一部分代码如下：

```python
#primal_attack2
def primal_attack2(A,b,m,n,p,esz):
    L = block_matrix(
        [
            [matrix(Zmod(p), A).T.echelon_form().change_ring(ZZ), 0],
            [matrix.zero(m - n, n).augment(matrix.identity(m - n) * p), 0],
            [matrix(ZZ, b), 1],
        ]
    )
    #print(L.dimensions())
    Q = diagonal_matrix([1]*m + [esz])
    L *= Q
    L = L.LLL()
    L /= Q
    res = L[0]
    if(res[-1] == 1):
        e = vector(GF(p), res[:m])
    elif(res[-1] == -1):
        e = -vector(GF(p), res[:m])
    s = matrix(Zmod(p), A).solve_right((vector(Zmod(p), b)-e))
    return s
```

<br/>

<br/>

#### Dual attack(最基础版本)

就像前文提到过的，primal attack找私钥s的基础在于样本本身需要是LWE样本，我们刚才的规约也都是建立在一个m=128，n=77的LWE样本上的。

然而，对于"crypto_sign_in_5"这个题目，要找到私钥s首先要找出四组m=32，n=77的LWE样本，然后将他们拼接才行。也就是说其实我们真正面对的是一个DLWE问题，这个时候就轮到Dual attack出场了。但是Dual attack依然做不到直接判定一个m=32，n=77的样本究竟是不是LWE样本，还是要碰运气，取四组都是LWE样本的才行。

(这也是为什么用优化过后的primal attack做这个题目就够，因为也是每次随机取四组，看有没有两次取的样本规约出相同的，那就是私钥s)

而Dual attack的基本原理其实并不复杂，依然先写出LWE样本的基础式子：
$$
As + e = b
$$
如果我们能在A的左核里找到一个短向量u，那么由于有(这里的0是n维零向量的意思)：
$$
uA = 0
$$
那么将上式均左乘一个u，得到：
$$
ue = ub \quad(mod\;p)
$$
由于u与e都是短向量，所以他们的内积应该是一个比较小的数字，所以计算模p下的ub的值，得到的应该是一个较小的数字。而如果不是LWE的样本的话，得到的ub可以看作是一个模p下的随机数，那么他的数量级期望应该是p/2。这就达到了区分出LWE样本的目的。

这一部分代码如下：

```python
#dual_attack1
def Dual_attack1(A,b,m,n,p):
    ker = Matrix(GF(p),A).left_kernel().basis()
    T = block_matrix(
        [
            [Matrix(ZZ,ker)],
            [identity_matrix(m)*p]
        ]
    )
    #print(T.dimensions())
    res = Matrix(ZZ,T).BKZ()[m-n]
    u = vector(GF(p),res)
    ub = u*vector(GF(p),b)
    if(ub > p//2):
        ub = p - ub
    #print(int(ub).bit_length())
    if(int(ub).bit_length() < int(p).bit_length()-5):
        return True
    else:
        return False
```

放在这个题里，其具体实现就可以描述成：随机选择四组样本，如果全都是LWE样本，那么Dual attack得到的值ub就会比较小(经测试183bit左右)，一旦发现满足要求的四组数据，就可以用前面的primal attack计算出私钥s了。

可以想见，Dual attack也是约束越多越好，m越大，约束越多，越不容易误判。对于这个题目的情况，正确的LWE样本的ub是183bit左右，而就算是纯随机数，也还是会有一定的概率落在这个范围以内(虽然并不高)。并且即使真的就是LWE样本，ub也可能大于这个范围。取五组虽然效果会好很多，但是就需要20多分钟才能规约出来，时间又太长了。

同时，这种最基本的Dual attack维数并不低，跑一组要两分多钟，虽然比起最基础的primal attack要快不少，但是比起优化过的primal attack就会显得比较慢。因此优化的方向依然是降维。

<br/>

<br/>

#### Dual attack(优化版本)

注意到，最基础的Dual attack建的矩阵为：
$$
\left(\begin{matrix}
kernel_{(m - n)\times m}\\
p\times I_{m \times m}
\end{matrix}\right)
$$
其中kernel是由m-n个左核空间中的基向量组成的矩阵，这样形成的矩阵是一个(2m-n)xm维的非方阵，因此维数大的同时，还需要在第m-n行及以后才能找到需要的短向量，并不是特别符合预期。

所以现在的优化目的有两个 ：

+ 降维
+ 把格变成方阵

然后就可以发现其实对primal attack的格优化思路依然可以用在这里。也就是对kernel矩阵求行最简形，使它变成：
$$
kernel_{(m - n)\times m} = [I_{(m - n)\times (m - n)},kernel'_{(m - n) \times n}]
$$
同样的道理，由于前面的m-n维只与单位阵做计算，因此得到的值都不会超过p，所以不需要+kp，因此下方p的单位阵就可以省下m-n维，这样格就变成了一个mxm的方阵了，就可以一分钟左右做一次Dual attack(实测发现用BKZ似乎效果会好一点)。虽然只取四组的话，误判的可能性依然存在。但由于降了维，可以测试出即使是取m=32x5，也只需要三分钟不到就可以做一次dual attack，就比未优化的版本快非常多了。

这一部分代码如下：

```python
#dual_attack2
def Dual_attack2(A,b,m,n,p):
    ker = Matrix(GF(p),A).left_kernel().basis()
    T = block_matrix(
        [
            [Matrix(GF(p),ker).echelon_form().change_ring(ZZ)],
            [matrix.zero(n, m-n).augment(matrix.identity(n) * p)]
        ]
    )
    #print(T.dimensions())
    res = Matrix(ZZ,T).BKZ()[0]
    u = vector(GF(p),res)
    ub = u*vector(GF(p),b)
    if(ub > p//2):
        ub = p - ub
    print(int(ub).bit_length())
    if(int(ub).bit_length() < int(p).bit_length()-5):
        return True
    else:
        return False
```

<br/>

<br/>

### 总结

没想到总结要说什么好，干脆贴个完整代码吧：

```python
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
from sage.crypto.lwe import LWE, samples

p = 0xfffffffffffffffffffffffffffffffeffffffffffffffff
m,n = 32*4,77
esz = 2**64
F = GF(p)
V = VectorSpace(F,n)
D = DiscreteGaussianDistributionIntegerSampler(esz)

#gen private_key s
lwe = LWE(n=n, q=p, D=D)
s = lwe._LWE__s
print(s)


#primal_attack1
def primal_attack1(A,b,m,n,p,esz):
    L = block_matrix(
        [
            [matrix.identity(m)*p,matrix.zero(m, n+1)],
            [(matrix(A).T).stack(-vector(b)).change_ring(ZZ),matrix.identity(n+1)],
        ]
    )
    #print(L.dimensions())
    Q = diagonal_matrix([p//esz]*m + [1]*n + [p])
    L *= Q
    L = L.LLL()
    L /= Q
    for res in L:
        if(res[-1] == 1):
            s = vector(GF(p), res[-n-1:-1])
            return s
        elif(res[-1] == -1):
            s = -vector(GF(p), res[-n-1:-1])
            return s


#primal_attack2
def primal_attack2(A,b,m,n,p,esz):
    L = block_matrix(
        [
            [matrix(Zmod(p), A).T.echelon_form().change_ring(ZZ), 0],
            [matrix.zero(m - n, n).augment(matrix.identity(m - n) * p), 0],
            [matrix(ZZ, b), 1],
        ]
    )
    #print(L.dimensions())
    Q = diagonal_matrix([1]*m + [esz])
    L *= Q
    L = L.LLL()
    L /= Q
    res = L[0]
    if(res[-1] == 1):
        e = vector(GF(p), res[:m])
    elif(res[-1] == -1):
        e = -vector(GF(p), res[:m])
    s = matrix(Zmod(p), A).solve_right((vector(Zmod(p), b)-e))
    return s


#dual_attack1
def Dual_attack1(A,b,m,n,p):
    ker = Matrix(GF(p),A).left_kernel().basis()
    T = block_matrix(
        [
            [Matrix(ZZ,ker)],
            [identity_matrix(m)*p]
        ]
    )
    #print(T.dimensions())
    res = Matrix(ZZ,T).BKZ()[m-n]
    u = vector(GF(p),res)
    ub = u*vector(GF(p),b)
    if(ub > p//2):
        ub = p - ub
    #print(int(ub).bit_length())
    if(int(ub).bit_length() < int(p).bit_length()-5):
        return True
    else:
        return False


#dual_attack2
def Dual_attack2(A,b,m,n,p):
    ker = Matrix(GF(p),A).left_kernel().basis()
    T = block_matrix(
        [
            [Matrix(GF(p),ker).echelon_form().change_ring(ZZ)],
            [matrix.zero(n, m-n).augment(matrix.identity(n) * p)]
        ]
    )
    #print(T.dimensions())
    res = Matrix(ZZ,T).BKZ()[0]
    u = vector(GF(p),res)
    ub = u*vector(GF(p),b)
    if(ub > p//2):
        ub = p - ub
    print(int(ub).bit_length())
    if(int(ub).bit_length() < int(p).bit_length()-5):
        return True
    else:
        return False



#gen sample(LWE or random)
sample = list(zip(*samples(m=m, n=n, lwe=lwe)))
#sample = list(zip(*[[V.random_element(),(F.random_element())] for _ in range(m)]))
A,b = sample


#test
#print(primal_attack1(A,b,m,n,p,esz))
#print(primal_attack2(A,b,m,n,p,esz))
#print(Dual_attack1(A,b,m,n,p))
#print(Dual_attack2(A,b,m,n,p))
```

<br/>

<br/>