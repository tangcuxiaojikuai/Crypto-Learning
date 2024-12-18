## 代数结构(algebraic structure)

#### 概念

$$
(G,*,+,...)
$$

+ G：**非空**集合
+ 一个或多个二元运算
+ 封闭性

#### 例子

非代数结构：$ (Z,\div)$

<br/>

<br/>

<br/>

## 群(group)

### 群

#### 概念

$$
(G,*)
$$

+ 封闭性
+ 结合律
+ 单位元(identity)：e
+ 逆元(inverse)

一般可只用集合名简写群，比如"G是个群"。

#### 例子

$(Z,+),(Q-\{0\},\times),(Z_p^*,\times)$

#### 性质

+ 群里单位元唯一
+ 群里每个元素逆元唯一
+ 消去律：$a*b = a*c \rightarrow b = c$
+ $(ab)^{-1} = b^{-1}a^{-1}$

+ $ (a^{-1})^{-1} = a$

#### 扩展概念

**有限群**：G是有限集合，G中元素个数称为G的阶，记为$|G|$

**无限群**：G是无限集合

**平凡群**：G中仅有单位元一个元素

**阿贝尔群(abelian group)**：满足交换律，又称交换群(commutative group)。证明是阿贝尔群仅需证明：$(a*b)^2 = a^2*b^2$

<br/>

<br/>

### 子群(subgroup)

#### 概念

若G是群，H是G的**非空子集**，若H也是**群**，则H是G的子群。

G的平凡子群：$(\{e\},*),(G,*)$

#### 例子

$(Z,+),(Q,+),(R,+)$，前面都是后面的子群

#### 判断子群

方法一：

+ H是G的非空子集
+ $\forall a,b \in H,ab^{-1}\in H$

方法二(对于有限子群H)：

+ H是G的非空子集，且H是有限集
+ H满足封闭性

#### 构造子群

+ 对于阿贝尔群G，$ G^m = \{a^m | a\in G\}$是G的子群。(即G中所有元素运算m次方，得到的集合构成群)例如：$(Z_p^*)^2$与$Z_p^*$，$3Z$与$Z$。

+ 对于阿贝尔群G，$ G\{m\} = \{a \in G | a^m = e\}$是G的子群。(即G中所有运算m次后得到e的元素，得到的集合构成群)

> 扩展：关于整数群及其子群的一些定理：
>
> 对于Z：
>
> + 若H是Z的子群，则存在唯一非负整数m，使得H=mZ
> + m1和m2是非负整数，则 $m2|m1 \leftrightarrow m1Z \subset m2Z$。例如：$27Z \subset 9Z \subset 3Z \subset Z$
>
> 对于Zn，性质类似：
>
> + 若H是Zn的子群，则存在n的唯一正因子m，使得$H = mZ_n$
> + m1和m2是n的正因子，则 $m2|m1 \leftrightarrow m1Z_n \subset m2Z_n$。例如：$9Z_{27}\subset 3Z_{27}$

+ 对于阿贝尔群G，若H1、H2都是G的子群，则 $H = H_1H_2 = \{a_1*a_2 | a_1 \in H_1,a_2 \in H_2\}$也是G的子群。(即阿贝尔群的两个子群的笛卡尔积构成的集合也是G的子群)
+ 对于阿贝尔群G，若H1、H2都是G的子群，则 $H = H_1 \cap H_2$也是G的子群。(即阿贝尔群的两个子群的交集也是G的子群)

> 扩展：
>
> + 对于阿贝尔群G，若H1、H2、...、Hn都是G的子群，则 $H = H_1 \cap H_2 ... \cap H_n$也是G的子群。(即阿贝尔群的多个子群的交集也是G的子群)

+ 对于任意群G，取其中一个元素a，则$<a> = \{a^z|z \in Z\}$是由a生成的G的子群。这个子群是个循环群。

<br/>

<br/>

### 陪集(coset)

#### 概念

若H是G的子群，则取G中任意元素a与H中的所有元素运算，得到一个集合(注意只是一个集合，并不一定是群)，这个集合就是H关于a在G中的陪集。

即(其实这里是左陪集，但是这里不讨论这个，默认左陪集等于右陪集)：
$$
[a]_H = \{a*h | h \in H,a \in G\}
$$
其中，a称为该陪集的代表元(representative)。

> 补充：就我个人而言，陪集可以简单理解为：根据G中元素a与子群H的不同关系，将a分进不同陪集，则G中所有元素都会进入到它对应的陪集里，并且所有陪集恰好两两不交、并集为G。这就构成G的一个划分。

#### 性质

+ $a \in [a]_H$，即每个元素会进入以自己为代表元的陪集中
+ $[e]_H = H$，即以单位元构造的陪集就是子群H的集合本身。这个陪集也叫平凡陪集

> 补充：这里也可以看出，除了平凡陪集外，其他的陪集都不是子群，因为陪集之间两两不交，而单位元仅在平凡陪集里

+ $\forall a \in H,[a]_H=H=[e]_H$，即如果用本身就在子群H中的元素构造陪集的话，这个陪集就是平凡陪集
+ $[a]_H = [b]_H \leftrightarrow a^{-1}b \in H$
+ 若a和b在同一个由H构造的陪集中，则可以写作：$a \equiv b \quad(mod\;H)$，这个二元关系的含义可以看作是说a、b与子群H的关系相同

#### 扩展概念

**拉格朗日定理**：由上面的若干性质，可以看出，由H构造的所有陪集大小均相等，且都等于H的大小。这就引出拉格朗日定理，即：

+ G是有限群
+ H的G的子群
+ $|H| \mid |G|$

也就是说，若G是有限群，则其任意子群的阶都是G的阶的因子。例如：若$|G|=15$，则其子群阶不可能在1，3，5，15以外的数产生。

**正规子群(normal subgroup)**：对于群G，如果由他的子群H构造的所有左右陪集均相等，那么H就是G的正规子群

<br/>

<br/>

### 商群

#### 概念

对于群G的正规子群H，定义由H产生的所有陪集构成的集合：
$$
G/H = \{[a]_H | a\in G\}
$$
并定义二元运算 * ：
$$
[a]_H * [b]_H = [a*b]_H
$$
则$(G/H,*)$就称作G模H下的商群。

> 补充：这里需要注意，H必须是G的正规子群，否则构不成商群

#### 性质

+ $[e]_H$是$G/H$的单位元，也就是商群的单位元是平凡陪集
+ 商群的阶记为$[G:H]$，等于$\frac{|G|}{|H|}$，此处需要G是有限群。其实也就等于由H构造的陪集的个数

若G是阿贝尔群，H是G的子群，则：

+ 任意元素与H构造的左右陪集均相等
+ H一定是正规子群
+ 由上一条可知，任意H都可以构造商群
+ 构造出的商群也是阿贝尔群

#### 例子

比如$Z/nZ$，其实就是Z模nZ下的商群，也就是$Z_n$，这个商群的集合就是模n下所有剩余类的集合。

<br/>

<br/>

### 群同态(group homomorphism)

#### 概念

设有两个群$(G,*)$和$(G',\otimes)$，记一个映射$f:G \rightarrow G'$，如果其满足对任意G中元素a、b，均有：
$$
f(a*b) = f(a)\otimes f(b)
$$
则f是G到G'的群同态。

#### 扩展概念

**单一同态**：f是单射

**满同态**：f是满射

**同态像(homomorphic image)**：指对于一个同态$f:G \rightarrow G'$，G'能在G中找到原像的所有元素的集合，记为$Im\;f$。简单来说就是G'中能够被映射到的元素构成的G'的子集。

> 补充：可以看出满同态就等价于 $Im\; f = G'$

**同态核(homomorphic kernel)**：指对于一个同态$f:G \rightarrow G'$，G中经映射后到达G'中单位元的所有元素构成的集合，记为$Ker\;f$

> 补充：
>
> + 若$f:G \rightarrow G'$是群同态，那么如果$f(a)=f(b)$，则 $a \equiv b \quad(mod\;Ker\;f)$，也就是说a和b在同一个由kernel生成的陪集中
> + 由上一条就可以知道，f是单一同态，等价于$Ker\;f = \{e\}$，也就是同态核中只有单位元

**嵌入映射(inclusion map)**：若H是G的子群，则对于同态$f:H \rightarrow G,f(a) = a$是一个嵌入映射。嵌入映射一定是单一同态。

**自然映射(natural map)**：记H是G的正规子群，则将G中任意元素映射进由H和该元素构成的陪集的映射就是自然映射。也就是说这个f是将G映射到G/H的映射。可以看出自然映射的两个性质：

+ 自然映射的同态核就是H本身，因为商群的单位元是平凡陪集，而只有H中的元素才会映射到平凡陪集
+ 自然映射一定是满同态

> 补充：雅可比映射：
>
> + 雅可比映射也就是将$Z_n^*$中的元素a映射成其与n的雅可比符号$(\frac{a}{n})$的映射。
> + 其kernel就是雅可比符号为1的元素集合，当n难以分解时，难以根据a的雅可比符号为1判定a是否是模n下的二次剩余
> + 但是，不在kernel中的元素一定是模n下的二次非剩余

#### 性质

+ $f(e) = e'$，即单位元同态后仍到达单位元
+ $f(a^{-1}) = f(a)^{-1}$，即一个元素的逆元同态后仍到达对应逆元
+ $f(a^m) = f(a)^m$

+ 若H是G的子群，则f(H)构成的集合是G'的子群。由此可以推出$Im\;f$是G'的子群(把H取作G本身即可)
+ $Ker \; f$是G的子群，且是正规子群

<br/>

<br/>

### 群同构(group isomorphism)

#### 概念

f既是单射又是满射，并且还是群同态，则f是群同构，可以记为：
$$
G \cong G'
$$
所有同构的群本质上都相同。

#### 扩展概念

**自同构(group automorphism)**：G和G自身的同构

**第一同构定理(first isomorphism theorem)**：又称基本同态定理(FHT,fundamental homomorphism theorem)，其内容为：设 $f:G \rightarrow G'$是群同态，那么有：
$$
G/Ker\;f \cong Im\;f
$$
这说明：**同态核构造的商群和同态像是同构的**。他表示的映射是，将由kernel构成的商群里的陪集，映射成其对应代表元的像。

<br/>

<br/>

### 循环群(cyclic group)

#### 概念

设G是群，若存在G中的元素g，使得(对这个符号不明白可以回看前面子群中构造子群部分)：
$$
G = < g > 
$$
则G是循环群，g是G的生成元(generator)

#### 扩展概念

**有限循环群、无限循环群**：顾名思义。

**元素的阶(order)**：设G是群(并不一定要是循环群)，对于G中元素a，如果n是满足$a^n=e$ 的最小正整数，则n是a的阶。

> 注意一定要是"最小"、"正"

**原根(primitive root)**：在$(Z_n^*,\times)$中，若元素g的阶等于$\phi(n)$，则g是模n下的原根。

#### 例子

+ $(Z,+)$是循环群，生成元是$\pm 1$
+ $(mZ,+)$是循环群，生成元是$\pm m$
+ $(Z_5^*,\times)$是循环群，2、3均是生成元

#### 性质

循环群本身：

+ 循环群一定是阿贝尔群
+ 循环群的子群一定是循环群
+ 循环群的商群一定是循环群。(由于循环群的子群一定是循环群，而循环群又一定是阿贝尔群，所以循环群一定是阿贝尔群，就一定是正规子群，所以循环群的所有子群都一定可以构造商群)

涉及到阶：

+ 有限循环群的阶是n，则生成元g的阶也是n，且 $g^k\;(k\leq n)$ 两两不同

+ 对所有 $d|n$，n阶有限循环群恰好包含一个唯一的d阶循环子群。

  > 例子：若有一个阶为15的循环群G，那么他恰有四个子群，阶分别为1、3、5、15，且均为循环群

+ n阶的有限循环群$<g>$，则 $g^k\;(k \in Z)$ 的阶为$\frac{n}{gcd(n,k)}$

+ 由上一条可以看出，当k与n互素时，g^k也是生成元。所以阶为n的有限循环群共有$\phi(n)$个生成元

+ G是n阶有限循环群，d是n的正因子，则G中阶为d的元素共有$\phi(d)$个。

  > 这可以由上方的性质推出来，因为取阶为d的元素作生成元，产生一个子群，该子群阶也就为d，而这个子群也是有限循环群，所以有$\phi(d)$个生成元

+ G是有限群(不一定是循环群)，则a的阶是G的因子。

  > 这是因为$<a>$是G的子群，而这个子群的阶就是a的阶，所以可以由拉格朗日定理推出本条

+ 由上一条可知，素数阶的群一定是有限循环群

涉及到同构：

+ 若两个群同构，其中一个群是循环群，则另一个也是循环群
+ 无限循环群都与$(Z,+)$同构
+ n阶有限循环群都与$(Z_n,+)$同构

涉及到原根(设p是奇素数，e是正整数)：

+ 只有当$n=1,2,4,p^e,2p^e$时，模n下才有原根

  > 这其实说明了很重要的一点，就是RSA算法中，模n下不会有原根，也就是没有阶为$\phi(n)$的元素，所有元素阶都只是$\phi(n)$的因子。这同时说明RSA的$(Z_n^*,\times)$不是循环群

+ 模n下的原根一共有$\phi(\phi(n))$个

  > 注意这里要求n必须有原根。由于原根存在的情况下，$(Z_n^*,\times)$是一个大小为$\phi(n)$的有限循环群，所以就有这条性质

+ 寻找原根：目前只有概率性算法，其中一个常见的算法是：随机生成一个数a，对他检查所有$\phi(n)$的素因子$p_i$，若均有$a^{\frac{\phi(n)}{p_i}} \neq 1$，则a是原根

<br/>

<br/>

<br/>

## 环(ring)

### 环

#### 概念

$$
(R,+,\times)
$$

+ $(R,+)$构成加法阿贝尔群

+ $(R,\times)$满足封闭性、结合律

  > 补充几个关于群的拓展定义：
  >
  > **广群(groupoid)**：满足封闭性
  >
  > **半群(semigroup)**：满足封闭性、结合律
  >
  > **幺半群(monoid)**：也称独异点，满足封闭性、结合律且有单位元
  >
  > 结合前文中的定义有包含关系：循环群$\subset$阿贝尔群$\subset$群$\subset$幺半群$\subset$半群$\subset$广群
  >
  > 因此，这一条也可以说成：$(R,\times)$构成半群

+ 乘法对加法满足分配律

一般也可以只用集合名简称环，如"R是个环"。

#### 例子

$(Z,+,\times)$，$(Z_n,+,\times)$，$(Z[x],+,\times)$

#### 扩展概念

**有限环、无限环**：顾名思义。

**含幺环(ring with identity)**：$(R,\times)$构成幺半群。

**交换环(commutative ring)**：满足乘法交换律的环。

**含幺交换环(commutative ring with identity)**：既是含幺环又是交换环的环。

**单位元**：若环是含幺环，则单位元一般特指乘法单位元，记为e。

**平凡环(trivial ring)**：$R = \{\theta\}$，即仅含零元的环，零元概念在下方介绍。

**非平凡环(non-trivial ring)**：$R \neq \{\theta\}$

<br/>

<br/>

### 零元(zero element)

#### 概念

$\times$是定义在非空集合A上的二元运算，若A中存在一个元素$\theta$，使得对A中任意元素a均有：
$$
a\times \theta = \theta \times a = \theta
$$
则称$\theta$为零元。

#### 性质

+ 群中无零元

  > 当群是大小为1的平凡群时，群中元素视为单位元，不视为零元；当群大小大于1时，如果存在零元，则零元无逆元，不符合群的定义

+ 环中零元即加法单位元
+ 环中除了零元外的所有元素，统称为非零元素

#### 扩展概念

**零因子(zero divisor)**：环R中，若有非零元素a、b使得：
$$
a\times b = \theta 
$$
则a和b为零因子。例如：环$Z_6$中的2、3是零因子。

> 之所以叫零因子，可以理解为：零元在环中可以分解为两个非零元素的乘积，因此这两个非零元素是零元的因子。

<br/>

<br/>

### 域(field)

#### 概念

若$(F,+,\times)$是一个域，则它满足以下条件：

+ $(F,+)$是阿贝尔群
+ $(F-\{\theta\},\times)$是阿贝尔群
+ 乘法对加法满足分配律

#### 扩展概念

**整环(integral domain)**：一个环满足下列条件即为整环：

+ 含幺交换环

+ $e \neq \theta$

  > 这一条等价于"非平凡环"

+ 无零因子

**除环(division ring)**：也称斜域(skew field)。若$(R-\{\theta\},\times)$是群，则R是除环。若$(R-\{\theta\},\times)$是阿贝尔群，则R是域。可以看出，乘法群满足交换律的除环就是域。

#### 性质

+ 域一定是整环
+ 无限整环不一定是域，比如$(Z,+,\times)$并不是域
+ 有限整环一定是有限域
+ 域是非平凡的(至少包含e和$\theta$两个元素)，且没有零因子
+ 可以看出以下包含关系：域$\subset$整环$\subset$含幺交换环$\subset$环

<br/>

<br/>

### 特征(characteristic)

#### 概念

R是环，如果存在最小正整数m，使得对于R中任意元素a，都有：
$$
ma=\theta
$$
则m是环R的特征。如果不存在这样的m，则环R的特征是0。

> 要注意是最小正整数。环R特征记为$Char \; R$

#### 例子

+ $Char\;Z = 0$
+ $Char\;Z_n = n$

#### 性质

+ 含幺交换环的特征要么等于0，要么等于其乘法单位元e的加法阶

  > 补充：加法阶的概念：
  >
  > R是环，对于R中的元素a，若存在最小的正整数k使得$ka = \theta$，则k是a的加法阶。否则称a是无限加法阶的元素。

  > 注意这里一定要是含幺交换环，因为含幺交换环才有单位元一说

+ 整环的特征等于0或素数

  > 因为整环中没有零因子，若特征是合数的话会产生零因子而矛盾

+ 域的特征等于0或素数；有限域的特征是素数

  > 因为域一定是整环。而有限域所有元素的加法阶都一定有限，所以e的加法阶一定有限，因此只能是素数

<br/>

<br/>

### 子环(subring)和扩环(extension ring)

#### 概念

若$(R,+,\times)$是环，S是R的非空子集，如果$(S,+,\times)$也是环，则S是R的子环，R是S的扩环。

> 可以类比子群的概念

#### 例子

+ 平凡环是任意环的子环
+ Q是R的子环，R是Q的扩环
+ mZ是Z的子环

#### 判断子环

最简化版本的判断如下：R是环，若S满足如下条件，则S是R的子环：

+ S是R的非空子集
+ 减法封闭性：对$\forall a,b \in S,a-b \in S$
+ 乘法封闭性：对$\forall a,b \in S,ab \in S$

#### 性质

+ 子环一定包含环的零元，却不一定包含环的单位元

+ 含幺交换环的子环不一定仍是含幺交换环

  > 就比如：Z是含幺交换环，mZ是他的子环，若m不等于1，则e不在mZ中，因此mZ不是含幺交换环

+ 整环的子环未必是整环

+ 域的子环未必是域

  > 这两个都可以用平凡环举反例，因为平凡环是任意环的子环，但是e和$\theta$相等，与整环和域的要求矛盾

  > 非平凡的例子可以举：Z是Q的子环，但Q是域而Z不是域

#### 扩展概念

**子域(subfield)和扩域(extension field)**：F是域，F'是F的子环，若F'也是域，则F'是F的子域，F是F'的扩域。

> 如：Q是R的子域，R是Q的扩域

<br/>

<br/>

### 理想(ideal)

#### 概念

全称"理想子环"，其定义为：$(R,+,\times)$是环，I是R的非空子集，如果$(I,+,\times)$满足：

+ 加法子群：$(I,+)$是$(R,+)$的子群

+ 乘法吸收律：$\forall r \in R,\forall a \in I \rightarrow ra \in I(ar \in I)$

  > 可以理解为：R中的元素只要与I中元素做乘法，就会被吸进I里

则称I是R的左(右)理想。

> 后续均只关注交换环，因此不讨论左右理想，均视为理想，也叫做双边理想(two-sided ideal)

#### 例子

+ mZ是Z的理想
+ mZn是Zn的理想

#### 判断理想

最简化版本的判断条件如下：$(R,+,\times)$是交换环，I是R的理想当且仅当$(I,+,\times)$满足：

+ 加法封闭性：$\forall a,b \in I,a+b \in I$
+ 乘法吸收律：$\forall r \in R,\forall a \in I \rightarrow ra \in I$

#### 性质

+ 任何理想都是子环

  > 都叫理想子环了

+ 理想一定包含环的零元，却不一定包含环的单位元

  > 因为理想一定是子环

+ 若R是环，I是R的理想，则$e\in I \leftrightarrow I = R$

  > 也就是说只要R的e在理想中，则理想就是环本身

+ 子环不包含环的单位元，并不代表子环没有单位元。事实上，环和子环的单位元没有任何关系

  > 环有单位元，子环可能没有；环没单位元，子环也可能有；环和子环都有单位元，单位元可能并不同

#### 扩展概念

设R是环，I是R的理想：

**零理想(zero ideal)**：$\{\theta\}$ (其他的都称为非零理想)

**单位理想(unit ideal)**：R本身

**平凡理想(trivial ideal)**：$\{\theta\}$和R (其他的都称为非平凡理想)

**真理想(proper ideal)**：$I \subset R$ (I是R的真子集)

**主理想(principal ideal)**：R是交换环，取R中元素a，则称aR(=Ra)是由a生成的R的主理想，记为$(a)$。

> 由于是交换环所以不区分左(右)理想

> 例子：
>
> + $(\theta) = \{\theta\}$
> + $(e) = R$，此时要求R是含幺交换环
> + mZ是Z的主理想

> 性质：
>
> + 环的理想包含元素a，一定包含由a生成的主理想。例如：$4 \in (2) \rightarrow (4)\subseteq (2)$

**素理想(prime ideal)**：I是R的真理想，对R中任意元素a、b，如果$ab \in I$，则有$a\in I$或$b \in I$，此时I称作R的素理想。

> 可以类比素数的性质：若$p \mid ab$，则$p\mid a$或$p \mid b$

> 例子：
>
> + 若m是素数，则mZ是素理想
> + 零理想也是Z的素理想
>
> 可以推知，当且仅当m为素数或0时，mZ是Z的素理想

> 注意，因为I必须是真理想，所以m=1时，mZ不是Z的素理想，因为mZ=Z是单位理想，不是Z的真理想

**极大理想(maximal ideal)**：M是R的真理想。除R之外，不存在任何包含M的理想，则称M是R的极大理想。

> 例子：
>
> + m是素数时，mZ是Z的极大理想
> + 零理想不是Z的极大理想

#### 扩展性质

+ 域没有非平凡理想(只有零理想和单位理想)

  > 也就是说域的极大理想是零理想

+ 零理想是含幺交换环的极大理想，当且仅当这个环是域

<br/>

<br/>

### 商环(quotient ring)

#### 概念

> 先进行一个简单的推导：R是环，I是R的双边理想。那么由理想的定义知，$(I,+)$是$(R,+)$的正规子群，因此可以用理想I构造商群，商群中的元素是陪集。此时，如果给商群中的元素再配搭上合适的乘法，就能构成一个环，叫做商环。
>
> 其乘法$\otimes$定义如下：
> $$
> [a]_I \otimes [b]_I = [a \times b]_I
> $$

由推导可知，商环$(R/I,\oplus,\otimes)$，又称R模I的剩余类环，其元素是陪集(也可以叫做剩余类)，运算定义如下：

+ 加法：$[a]_I \oplus [b]_I = [a+b]_I$
+ 乘法：$[a]_I \otimes [b]_I = [a \times b]_I$

#### 性质

+ 零元：$[\theta]_I = I$

+ 若R有单位元e，则商环也有单位元，是以e为代表元的陪集：$[e]_I$

+ I是素理想$\leftrightarrow R/I$是整环

+ I是极大理想$\leftrightarrow R/I$是域

+ M是极大理想，则M是素理想

  > 注意本条性质要求R是含幺交换环

<br/>

<br/>

### 环同态(ring homomorphism)

#### 概念

$(R,+,\times)$ 和$(R',\oplus,\otimes)$是环，$\forall a,b \in R$，函数$f:R\rightarrow R'$满足：

+ $f(a+b) = f(a)\oplus f(b)$

  > 这个可以看作是$(R,+)$到$(R',\oplus)$的加法群同态

+ $f(a\times b) = f(a)\otimes f(b)$

则称f是R到R'的环同态。

#### 扩展概念

**同态核**：$Ker\;f = \{a \in R | f(a) = \theta'\}$

> 注意这里关联的是值域的零元而非单位元

**同态像、嵌入映射、自然映射、单一同态、满同态**：可类比群同态。

#### 性质

+ 环同态是单射，则$Ker\;f = \{\theta\}$

  > 注意是只包含零元而不是单位元

+ 环同态是满射，则$Im \; f = R'$

+ $f(ka) = kf(a),k \in Z$

+ $f(a^k) = f(a)^k,k \in Z^+$

  > 若f是满同态且R是含幺环，则$k \in Z$

+ 若f是满同态且R是含幺环，则$f(e) = e'$

+ 环的理想对应同态像的理想

  > 注意并不一定是值域的理想

+ $Ker\;f$是R的理想，且是双边理想

  > 其实就可以类比群同态的同态核，因为群同态的同态核也可以证明一定是正规子群，一定可以构造商群；同理环同态的同态核是双边理想，就也一定可以构造商环

<br/>

<br/>

### 环同构(ring isomorphism)

#### 概念

基本概念可完全类比群同构，包括环自同构、(环)第一同构定理等。

#### 扩展概念

**单位(unit)**：R是环，对R中元素a，如果存在R中元素b，使得：
$$
ab = ba = e
$$
则称a为单位。

> 也就是环中存在乘法逆元的那些元素就叫单位，可以看出存在单位的环必然存在乘法单位元e

**单位群(group of units)**：R是环，称
$$
R^* = \{a|a \in R,a是单位\}
$$
为R的单位群。

> 也就是R中所有单位构成的乘法群，其单位元就是环的单位元e

> 例子：
>
> + $Z_n^*$是$Z_n$的单位群

#### 性质

+ 对于环同构$f:R\rightarrow R'$，若a是R的单位，则f(a)是R'的单位
+ 对于环同构，其单位群同构：$R^* \cong R'^*$

<br/>

<br/>

### 多项式环(polynomial ring)

#### 扩展概念

**环R上的多项式(polynomial)**：$f(x) = \sum_{i=0}^{k}a_ix^i = a_0 + a_1x + ... + a_kx^k$，其中系数和变量x均可以取环R中任意元素。

> + 系数：$a_i$
> + 常数项：$a_0$
> + 首项系数：$a_k$
> + 度：k，要求k是非负整数，记为$deg(f(x))$
> + 不定元(变量)：x

**常数多项式(constant polynomial)**：对于R中元素a，可以看作是只有常数项的多项式：
$$
a = a + \theta x + ... \theta x^k
$$
这种多项式称为常数多项式。

> 因此，环R可以看作是R[x]的子环，R[x]的零元也就是R的零元$\theta$，称为零多项式

#### 概念

**多项式环**：所有关于不定元x、系数属于环R的多项式
$$
f(x) = a_0 + a_1x + ... + a_kx^k,a_i \in R
$$
形成的环，称为R上的多项式环，记为$R[x]$，$R$叫$R[x]$的基环(base ring)。多项式环上的加法和乘法就是多项式的加法和乘法。

> 注意：$f(x)g(x)$在b中的取值不一定等于$f(b)g(b)$，这是因为R不一定是交换环。如果是交换环则一定满足上式

#### 性质

+ $deg(f(x)+g(x)) \leq max(deg(f(x)),deg(g(x)))$

+ $deg(f(x)g(x)) \leq deg(f(x)) + deg(g(x))$

+ 若R是整环，则$deg(f(x)g(x)) = deg(f(x)) + deg(g(x))$

  > 这是因为整环没有零因子，因此任意两个非零元素相乘都不为零元，因此多项式首项系数不等于零元

+ $R = \{\theta\} \leftrightarrow R[x] = \{\theta\}$

+ S是R的子环，当且仅当S[x]是R[x]的子环

+ R是含幺环，当且仅当R[x]是含幺环，且单位元均为e

+ R是交换环，当且仅当R[x]是交换环

+ R是整环，则R[x]是整环

+ F是域，则F[x]是整环

  > 注意F[x]只是整环，且**一定不是域**。这是因为如果是域的话，F[x]中的多项式f(x)就一定有逆元g(x)，此时$f(x)g(x)=e$，说明除了常数项以外的系数都是零元，但是整环是没有零因子的，因此矛盾

<br/>

<br/>

### 域上多项式

#### 概念

**多项式除法(polynomial division)**：设域F，F是F[x]的基域，若$f(x),g(x) \in F[x],g(x) \neq \theta$，则存在唯一的$q(x),r(x) \in F[x]$，使得：
$$
f(x) = q(x)g(x) + r(x)\;,\;deg(r(x)) < deg(g(x))
$$

> + 被除式：f(x)
> + 除式：g(x)
> + 商式：q(x)
> + 余式：r(x)
> + 若$r(x) = \theta$，则称f(x)是g(x)的倍式

**首一多项式(monic polynomial)**：设R是含幺环，$f(x) \in R[x],f(x)$的首项系数是单位元e，则f(x)是首一多项式。

> 之所以R必须是含幺环，是因为不是所有环都有单位元e。而由于域一定有单位元，所以域有首一多项式这个概念。

**最大公因式**：设域F，$f_1(x) , f_2(x),...,f_n(x) \in F[x]$，且f(x)不全为$\theta$，则存在唯一的首一多项式$d(x) \in F[x]$，使得：

+ $d(x) \mid f_1(x) , d(x) \mid f_2(x) , ... , d(x) \mid f_n(x)$
+ $\forall c(x) \in F[x]$，若也有$c(x) \mid f_1(x) , c(x) \mid f_2(x) , ... , c(x) \mid f_n(x)$，则$c(x) \mid d(x)$

则d(x)称为$f_1(x) , f_2(x),...,f_n(x)$的最大公因式。

> 注意d(x)必须是首一多项式

> 如果d(x)为e，则称$f_1(x) , f_2(x),...,f_n(x)$互素

> 基本可以完全类比数论中的最大公约数，比如可以应用欧几里得算法求最大公因式

**最小公倍式**：同样可类比数论中的最小公倍数，需要注意的是最小公倍式依然必须是首一多项式。

<br/>

<br/>

### 多项式环的理想

#### 扩展概念

**主理想整环**：理想都是主理想的整环，称主理想整环。

#### 性质

+ 设$a \in R$，则$J = \{g(x) \in R[x] \;|\; g(a) = \theta\}$是R[x]的理想

  > 因为显然任何多项式与$J$中多项式相乘就都会以a为根，满足乘法吸收律

+ 对于域上多项式环来说，F[x]一定是主理想整环。也就是对F[x]的每一个非零理想$J$，都存在唯一的首一多项式$g(x) \in F[x]$，使得$J = (g(x))$

<br/>

<br/>

### 不可约多项式(irreducible polynomial)

#### 概念

设域F，$p(x) \in F[x]$，如果$deg(p(x)) > 0,p(x) = b(x)c(x)$，则$deg(b(x)) = 0$或$deg(c(x)) = 0$，则称p(x)在F上不可约，又称在F[x]里不可约的或素的。

> 也就是不能分解为两个非常数多项式的多项式的乘积。若分解为常数多项式与度等于自身的多项式的乘积，这种分解称为平凡分解

> 注意只有在域上，才有不可约多项式的概念

#### 性质

对$\forall g(x)\in F[x],deg(g(x))>0 $，则g(x)可唯一地写成如下形式：
$$
g(x) = ap_1^{e_1}(x)p_2^{e_2}(x)...p_k^{e_k}(x)
$$
其中，$a \in F,p_1(x),p_2(x)...p_k(x) \in F[x]$ 是彼此不同的首一不可约多项式，$e_1,e_2,...,e_k \in N$。

> 注意是首一不可约多项式

> 可以类比数论中的算数基本定理，一个整数可以唯一性的分解成多个素数幂次的乘积。在这里不可约多项式就类似于素数的作用。

<br/>

<br/>

### 代数扩张

#### 概念

**扩域**：K是F的子域，F就是K的扩域或扩张(extension)。

> 子域和扩域的概念在之前介绍子环时已经扩展介绍过

**真子域**：K还是F的真子集，则K是F的真子域。

**素域(prime field)**：没有任何真子域的域。

> 即：素域的任何子域都是他本身

> 例子：Q是素域

#### 性质

+ 域F所有子域的交集K构成他的素子域

+ 设K是F的子域，F还有子集M，记K(M)是F所有包含K和M的子域的交集，则有：

+ + K(M)是K的扩域
  + K(M)是包含K和M的最小子域

  > M并不需要是子域，是F的子集即可

#### 扩展概念

**单扩域(simple extension field**)：如果M仅包含一个元素a，则记K(M)=K(a)，K(a)称为K的单扩域，a叫做K上K(a)的定义元素(defining element)。

**域上代数的**：K是F的子域，对于F中的元素a，存在非常数多项式f(x)，其系数来自K，有$f(a)= \theta$，称a是K上代数的。

> 即存在某个系数均来自K的多项式，使得a是这个多项式的根

**域上超越的**：K是F的子域，对于F中的元素a，对任何系数来自K的非常数多项式f(x)，都没有$f(a)= \theta$，称a是K上超越的。

> 即不存在某个系数均来自K的多项式，使得a是这个多项式的根

**代数扩域(algebraic extension field)**：若K是F的子域，L是K的扩域，若$\forall b \in L$是K上代数的，则称L是K的代数扩域或代数扩张(algebraic extension)。

<br/>

<br/>

### 极小多项式(minimal polynomial)

#### 概念

$J = \{f(x) \in K[x]\;|\; f(a) = \theta\}$，存在唯一的首一多项式$g(x) \in K[x]$，有$J = (g(x))$。则g(x)称为K上a的极小多项式(或定义多项式、不可约多项式)。

> 即：如果a是K上代数的，所有系数来自K的以a为根的多项式构成的集合J，等于某个唯一的首一多项式g(x)生成的主理想，这个g(x)就叫做K上a的极小多项式

> 极小多项式的概念必须与对应的子域K以及域中的某个具体元素a关联

#### 性质

+ g(x)一定是K[x]里的不可约多项式
+ $g(x) | f(x) \leftrightarrow f(x) \in K[x],f(a) = \theta$

+ g(x)是K[x]中以a为根的首一多项式中度最小的

<br/>

<br/>

### 域的扩张

#### 扩展概念

**向量空间(vector space)**：设V是非空集合，V中元素称为向量(vector)，并且元素都建立在域K上，K称为其基域(base field)，K中元素称为标量(scalar)，则称V是K上的向量空间。

> 例子： 
>
> + n维欧几里得空间$R^n$
> + 实系数度不超过n的全体多项式
> + 以实数为定义域和值域的全体连续函数

**向量空间的运算要求**：

对$\forall X,Y \in V,\forall a,b \in K,e \in K$，

+ 加法阿贝尔群：向量间的加法
+ 标量乘法：封闭性$aX \in V$、结合律$a(bX) = (ab)X$、单位元$eX = X$
+ 标量和向量之间彼此满足乘法对加法的分配律：$a(X+Y) = aX + aY,(a+b)X = aX + bX$

**基**：最大线性无关向量组。

**向量空间维度**：基里向量的个数。

#### 扩展性质

+ 域是其任意子域上的向量空间

  > 因此，扩域之于子域，可以完全等价于向量空间之于基域

+ K上F的度：将扩域和子域看作向量空间和基域，则K上F的度就是向量空间维度，记为$[F:K]$

#### 概念

**有限扩张(finite extension)**：设F是K的扩域，若F是K上有限维度的向量空间，则称K是F的有限扩张或有限扩域(finite extension field)。

> 注意有限指的是维度有限，并不指扩域中的元素个数有限。这等价于说有限域的度是有限的。

> 例子：
>
> + $[Q(\sqrt{2}):Q)] = 2$，其中$Q(\sqrt{2}) = \{a + b\sqrt{2} | a,b \in Q\}$，基为$\{1,\sqrt{2}\}$
> + $[C:R] = 2$，基为$\{1,i\}$
> + $[R:Q] = \infty$

#### 性质

+ $[k:K] = 1$
+ $[F:K] = 1 \leftrightarrow F = K$

> 即每个域都可以看做自身的扩域，且都是度为1的有限扩张

+ F是L的有限扩张，L是K的有限扩张，则F是K的有限扩张，且$[F:K] = [F:L][L:K]$
+ 有限扩张一定是代数扩张

<br/>

<br/>

### 单代数扩张(simple algebraic extension)

#### 扩展性质

+ $g(x) \in K[x]$是不可约多项式，则$(g(x))$是极大理想

  > 然后由前面商环的知识就可以知道，域上的多项式环与他的不可约多项式构造的商环是一个域，这一点非常重要

#### 概念

F是K的扩域，$a \in F$，且a是K上代数的，则称K(a)是K的单代数扩张或单代数扩域(simple algebraic extension field)。

#### 性质

+ $K(a) \cong K[x]/(g(x))$

+ $[K(a):K] = n$，n是极小多项式g(x)的度，且$(e,a,...,a^{n-1})$是K上K(a)的基

+ $\forall b \in K(a)$都是K上代数的，相应极小多项式的度都是n的因子

+ 若$g(x) \in K[x]$是K上的不可约多项式，则存在K的单代数扩张，他以g(x)的根为定义元素

  > 有这一条性质，构造单代数扩张就不再需要扩域F本身

+ $g(x) \in K[X]$是K上不可约多项式，a，b是g(x)的两个根，则$K(a) \cong K(b)$

#### 构造

举一些例子如下：

一、R是Q的扩域，$\sqrt{2} \in R$，

+ $Q(\sqrt{2})$是Q的单代数扩张。这是因为$\sqrt{2}$是Q上代数的，因为存在系数来自Q的多项式$g(x) = x^2-2$，其根是$\sqrt{2}$。

+ $[Q(\sqrt{2}):Q] = 2$，因为g(x)是Q上$\sqrt{2}$的极小多项式

+ 基的元素是$(e,\sqrt{2})$

二、$Z_3 = \{0,1,2\}$是域，$g(x) = x^2 + x + 2 \in Z_3[X]$是$Z_3$上不可约的，那么由上述性质有：

+ $Z_3[X]/(g(x))$是域
+ 设a是g(x)的根，则a是$Z_3$上代数的，则有以g(x)为极小多项式的单代数扩张$Z_3(a)$，代表元素是a
+ 由于g(x)度为2，所以$Z_3(a)$的基有两个元素$(e,a)$

总结一下，要构造K上的单代数扩张，其步骤为：

+ 找K上的不可约多项式g(x)

+ 找g(x)的一个根，设为a，那么就有单代数扩张K(a)

  > 不用管究竟是哪个根，因为由上面的最后一条性质可以知道，所有g(x)的根构造的单代数扩张都同构

+ 这个单代数扩张其实也就是商环$K[x]/(g(x))$

#### 应用

+ F是K的扩域，$a,b \in F$，则$K(a,b) = K(a)(b)$

+ F是K的扩域，$a_1,a_2,...,a_m \in F$，则$K(a_1,a_2,...,a_m) = K(a_1,a_2,...,a_{m-1})(a_m)$
+ 若$a_1,a_2,...,a_m $是K上代数的，则$K(a_1,a_2,...,a_m)$是有限扩张
+ L是K的有限扩张 $\leftrightarrow$ 存在$a_1,a_2,...,a_m\in L$是K上代数的，使得$L = K(a_1,a_2,...,a_m)$

#### 扩展概念

**分裂域(splitting field)**：对非常数多项式$f(x) \in K[X]$，同时包含K和f(x)的所有根的最小扩域L，称为K上f(x)的分裂域。

> 例如：对于$f(x) = 3(x^2-2)(x^2-3)(x^2-6)(x^2+1) \in Q[x]$，求f(x)的分裂域L。
>
> 由于$f(x) = 3(x-\sqrt{2})(x+\sqrt{2})(x-\sqrt{3})(x+\sqrt{3})(x-\sqrt{6})(x+\sqrt{6})(x-i)(x+i)$，所以有$L = Q(\sqrt{2}-\sqrt{2},\sqrt{3},-\sqrt{3},\sqrt{6},-\sqrt{6},-i,i) = Q(\sqrt{2},\sqrt{3},i)$
>
> 可以看出对对任意非常数多项式$f(x)\in F[X]$，都存在K上f(x)的分裂域，并且任何K上f(x)的分裂域均同构。

> 对于分裂域有以下性质：
>
> + 分裂域一定是有限扩张，因为f(x)的所有根都是K上代数的
> + $[L:K] \leq deg(f(x))!$

<br/>

<br/>

<br/>

## 有限域

### 有限域

#### 概念

元素个数(阶)有限的域。

> 回顾概念：
>
> + 特征：设有一个最小正整数m，使得对任意环R中元素a都有$ma = \theta$，则环R的特征为m，记为$Char\;R=m$，若不存在这样的m，则$Char \; R = 0$
>
>   > 有限域的特征一定是素数
>
>   > 特征也就是$me = \theta(e \in R)$的最小正整数m，若不存在这样的m则$Char \; R = 0$
>
> + 素域：不包含任何真子域的域

#### 性质

+ 阶为素数有限域必是素域，其特征是这个素数

+ 任何有限域都包含一个同构于$Z_p$的子域

+ 有限域的阶必定具有$p^n$的形式。p是有限域F的特征，也是其包含的素子域K的特征以及元素个数；n是F在其素子域K上的扩张维度

+ 对于任意素数p及正整数n，都存在阶为$p^n$的有限域

+ 任何阶为$p^n$的有限域都同构于$x^{p^n}-x$在$Z_p$上的分裂域

  > 简单描述一下这个结论的证明：首先令$f(x) = x^{p^n}-x$，则求他的导数为$f'(x) = p^nx^{p^n-1}-1$，在模p下就恒等于-1。而导函数没有零点说明方程没有重根，因此f(x)就有$p^n$个不同的根，F就有$p^n$个不同元素

+ 具有相同数量元素的有限域都同构，对于含q个元素的有限域，就可以统一表示为$F_q$或$GF(q)$

#### 扩展性质

+ **子域准则(subfield criterion)**：设$q = p^n$，有限域$F_q$的每个子域的阶都有$p^m$的形式，其中m是n的正因子。反之，如果m是n的正因子，则必然存在$F_q$的唯一子域，其阶为$p^m$。

  > 例如：$F_{2^{30}}$的子域有：$F_{2},F_{2^{2}},F_{2^{3}},F_{2^{5}},F_{2^{6}},F_{2^{10}},F_{2^{15}},F_{2^{30}}$

+ 有限域的乘法群一定是循环群，这个循环群的生成元称为$F_q$的本原元(primitive element)

  > 这就是sus那个题目需要用到的一个很重要的性质，有了这个性质那个题目就好解释多了

+ 若有限域有q个元素，则其乘法群有q-1个元素，有$\phi(q-1)$个本原元

+ $F_r$是$F_q$的有限扩张，则$F_r$是$F_q$的单代数扩张：对于任意$F_r$的本原元a，都有$F_r = F_q(a)$

  > 这个性质也就可以说成：任何一个有限域，都可以看作是他的本原元邻接他的一个子域形成的单代数扩张

+ 对于任意有限域$F_q$，任意正整数n，$F_q[X]$中都存在一个度为n的不可约多项式

  > 这个可以利用上一条性质证明。因为$F_{q^n}$是$F_q$的扩张维度为n的有限扩张，取他的一个本原元a构造单代数扩张就有：$F_{q^n} = F_q(a)$。这也就说明，a在$F_q$上的极小多项式，就是$F_q[X]$里的一个度为n的不可约多项式

#### 构造

这里主要讲到两种构造方法，一种基于不可约多项式构造商环，一种基于不可约多项式的根构造单代数扩张。二者都需要用到不可约多项式。

+ $f(x) \in F_q[X]$是不可约多项式，则$F_q[X]/(f(x))$是有限域。若$deg(f(x)) = n$，则$F_q[X]/(f(x)) = F_{q^n}$

  > 这也是sus那个题目需要用到的核心性质

+ $f(x) \in F_q[X]$是不可约多项式，算出f(x)的一个根a，则单代数扩张$F_q(a)$是有限域。若$deg(f(x)) = n$，则$F_q(a) = F_{q^n}$，其基为$e,a,...,a^{n-1}$

<br/>

<br/>

### 分圆域(cyclotomic field)

#### 概念

**分圆域**：n是正整数，K是任意域，$f(x) = x^n -e \in K[X]$在K上的分裂域就称为K上n次分圆域，记为$K^{(n)}$。

**n次单位根(nth roots of unity)**：记多项式$x^n - e$在$K^{(n)}$里的根为$b_i(1 \leq i \leq n)$，这些根称为K上n次单位根，其组成的集合记为$E^{(n)}$。

> 例如：n次分圆域$Q^{(n)}$就是$x^n-1$在Q上的分裂域，这是复数域的一个子域

#### 性质

+ $E^{(n)}$的结构依赖于n和K的特征m。若m不是n的因子，则$E^{(n)}$构成阶为n的乘法循环群，有$\phi(n)$个生成元。这些生成元称为K上n次本元单位根(primitive nth root of unity)

  > 这里要区分一下本原元、原根、n次本元单位根这三个概念，具体来说，三者的相同点是都是循环群的生成元。然而本原元是有限域乘法群的特有概念，原根是模运算下循环群的特有概念，n次本元单位根是特指$E^{(n)}$的生成元。不过这只是概念上的区分，三者的性质其实没什么差别。

<br/>

<br/>

### 分圆多项式(cyclotomiac polynomial)

#### 概念

若$E^{(n)}$是个乘法循环群(也就是说域K的特征m不是n的因子)，记$a_1,a_2,...,a_r$是K上所有n次本原单位根，则多项式：
$$
Q_n(x) = \prod_{i= 1}^{r}(x-a_i)
$$
就称为K上n次分圆多项式，其中$r = \phi(n)$。

> 也就是K上所有n次本原单位根构成的一次因式的乘积多项式，就是K上n次分圆多项式

> 注意只有在域K的特征m不是n的因子时，才有分圆多项式这一说。否则$E^{(n)}$并不是乘法循环群，也就不存在n次本原单位根

#### 性质

+ $deg(Q_n(x)) = \phi(n)$

+ $x^n - e = \prod_{d|n}{Q_d(x)}$

+ 若d是正整数n的真因子($1\leq d < n$)，则$Q_n(x) | \frac{x^n-e}{x^d-e}$

  > 显然$Q_n(x)$是$x^n-e$的因式，而$Q_n(x)$又与$x^d-e$互素(因为$x^d-e$的根都不是$E^{(n)}$里的n阶元素，而$Q_n(x)$的根都是$E^{(n)}$里的n阶元素，所以二者没有公共根，也就没有公因式)

<br/>

<br/>

### 伴随矩阵(companion matrix)

#### 概念

设K是域，$f(x) = a_0 + a_1x + ... + a_{n-1}x^{n-1} + x_n \in K[X]$是首一多项式，则f(x)的伴随矩阵是如下的nxn矩阵：
$$
A
=
\left(
 \begin{matrix}
 \theta & \theta & \theta & \cdots & \theta & -a_0 \\ 
 e & \theta & \theta & \cdots & \theta & -a_1 \\ 
 \theta & e & \theta & \cdots & \theta & -a_2 \\ 
 \vdots & \vdots & \vdots & \cdots & \vdots & \vdots \\
 \theta & \theta & \theta & \cdots & e & -a_{n-1} \\ 
  \end{matrix}
  \right)
$$

#### 性质

+ A是f(x)的根，即$f(A) = a_0I + a_1A + ... + a_{n-1}A^{n-1} + A^n = \theta$

+ 可以基于伴随矩阵A构造有限扩域(多项式度为n，则扩域大小为p^n)。基为：$I,A,...,A^{n-1}$，扩域$F_{p^n}$的元素是$b_0I + b_1A + ... + b_{n-1}A^{n-1},b_i \in F_p$

  > 例如：$f(x) = x^2 + e \in F_3[X]$，则伴随矩阵A为：
  > $$
  > A
  > =
  > \left(
  >  \begin{matrix}
  >  \theta & -e \\
  >  e & \theta
  >   \end{matrix}
  >   \right)
  >  =
  > \left(
  >  \begin{matrix}
  >  \theta & 2e \\
  >  e & \theta
  >   \end{matrix}
  >   \right)
  > $$
  > 扩域为$F_{3^2}$，元素有：$\theta , I,A,2I,2A,I+A,2I+A,I+2A,2I+2A$

+ 还有一种方法基于伴随矩阵构造有限域，这种方法需要利用分圆域。

  > 不写出来的原因是还没太看懂。。

<br/>

<br/>

### 多项式的阶(order)

#### 概念

又称多项式的周期(period)、多项式的指数(exponent)。设$f(x)$是有限域的非零多项式，阶需分两种情况考虑：

+ $f(x)$的常数项不为零元：$f(x)$的阶是使$f(x) | (x^n-e)$的最小整数n，记为$ord(f(x))$

+ $f(x)$的常数项为零元：设$f(x) = x^tg(x)$，其中$g(x)$的常数项不为零元，则$ord(f(x)) = ord(g(x))$

  > 即把含x的公因式提出来后，剩下的常数项不为零元的多项式的阶，就是原多项式的阶

#### 性质

+ 任意f(x)的阶都存在

+ f(x)的度不会大于f(x)的阶

+ 常数多项式以及多项式x的阶都为1

+ $g(x) \in F_q[X]$是不可约多项式，度为m，阶为n，常数项不为零元。那么有$a \in F_{q^m}^*,g(a) = \theta$，则n等于a在扩域$F_{q^m}$的乘法群的阶

  > 可以推出$ord(g(x)) | (q^m - 1)$

+ $g(x) \in F_q[X]$是不可约多项式，阶为n，常数项不为零元，则$ord(g(x)^b) = np^t$，其中p是$F_q$的特征，t是满足$p^t \geq b$的最小整数

#### 计算

+ 将f(x)分解为首一不可约多项式的乘积

  > 注意分解形式为$f(x) = ag_1^{b_1}(x)...g_k^{b_k}(x)$，也就是相同的多项式要看作同一个多项式参与后续计算

+ 分别计算这些首一不可约多项式的阶

+ 求这些阶的最小公倍数，就得到f(x)的阶

> 其实就和RSA中$\lambda(n)$的计算一样

<br/>

<br/>

### 本原多项式(primitive polynomial)

#### 概念

