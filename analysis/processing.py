import re

def parse_txt_file(file_path):

    image_names, polynomials = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    
    for line in lines:
        parts = line.split('.pngis:')
        # print(parts)
        image_name = parts[0]
        image_names.append(image_name)
        polynomial = parts[1]
        polynomial = polynomial.split(',')[0]
        print(f'parsing polynomials: {polynomial}')
        polynomials.append(polynomial)


    return image_names, polynomials

file_path = 'Terminal_oneline.txt'
# with open(file_path, 'r', encoding='utf-8') as file:
#     lines = file.readlines()

all_coefficients = []
image_names, polynomials = parse_txt_file(file_path)

print('data gained')
for polynomial in polynomials:
    print(f'polynomial: {polynomial}')
    coefficients = [0] * 9
    terms = polynomial.split('+')
    # print(f'terms: {terms}')
    for term in terms[:-1]:
        # print(f'processing term {term}')
        if r't@' not in term and 't' in term:
            term = term.replace('t', r't@1')
            print(f'added t@1 to term {term}')
            print(f'processing term {term}')

        coef, power = term.split(r't@')
        if coef == '':
            # print(f'coef of {term} is empty')
            coef = 1
        # 如果coef存在*，把*去掉
        elif coef == '-':
            coef = -1
        else:
            # print(f'coef of {term} is {coef}')
            coef = re.sub(r'\*', "", coef)
            # print(f'removed coef of {term} is {coef}')
        coefficients[int(power)] = int(coef)

    
    coefficients[0] = int(terms[-1])
    print(coefficients)
    # print(f"{image_name}: {polynomial}")

    all_coefficients.append(coefficients)

name_and_coefficients = list(zip(image_names, all_coefficients))
print(name_and_coefficients)

# save as csv
import csv
with open('Terminal_oneline.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(name_and_coefficients)
print('done')