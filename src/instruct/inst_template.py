
emsa_cot_template = '''As a metaphor sentiment specialist, analyze ONLY the sentiment interaction between source and target components. Ignore contextual sentiments. Follow these steps:

1. Metaphor Mapping: Identify how [source] characterizes [target]
2. Sentiment Transfer: Determine which affective values transfer between domains
3. Polarity Judgment: Classify final polarity (positive/negative/neutral)

Example:
{demonstration}

Now analyze:
Input: [{input}] | Source component: [{source}] | Target component: [{target}]
Your output: '''

emsa_icl_template = '''Analyze how the source component’s sentiment affects the target in this metaphor. Output ONLY polarity (positive/negative/neutral).

Example:
{demonstration}

Now analyze:
Input: [{input}] | Source component: [{source}] | Target component: [{target}]
Output: '''

emsa_mtl_template = '''Identify the source and target components in this metaphor. Output ONLY in the format "Source is [] | Target is []".

Example:
{demonstration}

Now identify:
Input: [{input}]
Output:'''

emsa_template = '''Analyze how the source component’s sentiment affects the target in this metaphor. Output ONLY polarity (positive/negative/neutral).
Input: [{input}] | Source component: [{source}] | Target component: [{target}]
Output: '''

cmsa_icl_template = '''作为一名隐喻情感专家，请分析隐喻结构（即隐喻源成分和目标成分对）的情感极性（正向/负向/中性）。

示例：
{demonstration}

现在请分析：
输入：[{input}] | 源成分：[{source}] | 目标成分：[{target}]
你的输出是？'''

cmsa_template = '''作为一名隐喻情感专家，请分析隐喻结构（即隐喻源成分和目标成分对）的情感极性（正向/负向/中性）。
输入：[{input}] | 源成分：[{source}] | 目标成分：[{target}]
你的输出是？'''

cmsa_cot_template = '''作为一名隐喻情感专家，请分析隐喻结构（即隐喻源成分和目标成分对）的情感极性。遵循以下步骤：
1. 隐喻映射：确定 [源] 如何描述 [目标]
2. 情感转移：确定哪些情感价值在不同领域之间转移
3. 极性判断：对最终极性（正向 / 负向 / 中性）进行分类

示例：
{demonstration}

现在请分析：
输入：[{input}] | 源成分：[{source}] | 目标成分：[{target}]
你的输出是？'''