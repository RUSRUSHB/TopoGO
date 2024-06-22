import re
polynomial_str = "t@4+-2*t@3+3*t@2+-3*t+2"
# coef, power = term.split('*t@')


coefficients = [0] * 9

terms = polynomial_str.split('+')
print(f'terms: {terms}')
for term in terms[:-1]:
    print(f'processing term {term}')
    if r't@' not in term and 't' in term:
        term = term.replace('t', r't@1')
        print(f'added t@1 to term {term}')
        print(f'processing term {term}')

    coef, power = term.split(r't@')
    if coef == '':
        # print(f'coef of {term} is empty')
        coef = 1
    # 如果coef存在*，把*去掉
    else:
        # print(f'coef of {term} is {coef}')
        coef = re.sub(r'\*', "", coef)
        # print(f'removed coef of {term} is {coef}')
    coefficients[int(power)] = int(coef)
coefficients[0] = int(terms[-1])
print(coefficients)