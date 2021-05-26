from typing import Dict


class Calculator:

    def calculate(self, numbers: Dict[int, int]):
        result = ""
        sorted_keys = sorted(numbers.keys())
        if 10 not in numbers.values() and 11 not in numbers.values():
            for e in sorted_keys:
                result += self.check_if_sign(numbers[e])
            return result
        filtered_signs = list(map(lambda x: numbers[x], list(filter(lambda key: numbers[key] >= 10, sorted_keys))))
        filtered_numbers = list(map(lambda x: numbers[x], list(filter(lambda key: numbers[key] < 10, sorted_keys))))
        for s in filtered_signs:
            if s == 10:
                num1 = filtered_numbers.pop(0)
                num2 = filtered_numbers.pop(0)
                filtered_numbers.insert(0, num1 + num2)
            if s == 11:
                num1 = filtered_numbers.pop(0)
                num2 = filtered_numbers.pop(0)
                filtered_numbers.insert(0, num1 - num2)
        return filtered_numbers.pop(0)

    def check_if_sign(self, element):
        if element == 10:
            return '+'
        if element == 11:
            return '-'
        return str(element)