from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)


        chars = sorted(set(text))

        stoi = {}

        for index, char in enumerate(chars):
            stoi[char] = index

        itos = {}

        for char, index in stoi.items():
            itos[index] = char

        print(stoi)
        print(itos)

        return (stoi, itos)


    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping

        result = []

        for i in text:

            result.append(stoi[i])

        return result


    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping

        result = []

        for i in ids:
            result.append(itos[i])

        return "".join(result)

       
