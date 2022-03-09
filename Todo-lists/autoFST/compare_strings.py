"""
"""


def lcs(s1, s2):
    """Modifies the original answer on
    https://stackoverflow.com/questions/48651891/longest-common-subsequence-in-python
    """
    indices = []

    matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                # print(i, j)
                indices.append((i, j))
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i - 1][j - 1] + s1[i]
            else:
                matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1], key=len)

    cs = matrix[-1][-1]

    return len(cs), cs, indices


def find_morphemes(s1, s2):
    """Ideally cuts strings into morphemes..."""
    # find consecutive is and js (or how does lcs break into segments)
    # max_len as below is misleading, e.g. e is unwanted in "steigen", "stiegen"
    # --> "st" "e" "gen" then discard "e"
    # cuts string by "st" and "gen"
    # --> "st" "ei" "gen"; "st", "ie", "gen"
    max_len, _, indices = lcs(s1, s2)
    results = []
    i = 0
    for i in range(len(indices)):
        print(i)
        print("====")
        print(indices[i])
        print(construct_next(indices[i]))
        cnt = 0
        found = []
        while construct_next(indices[i]) in indices:
            if indices[i] not in found:
                found.append(indices[i])
                cnt += 1
            if construct_next(indices[i]) not in found:
                found.append(construct_next(indices[i]))
            else:
                break
            i += 1
            cnt += 1
            print("found:")
            print(len(found))
            print("max_len:")
            print(max_len)
            print("cnt:")
            print(cnt)
            results.append((cnt, found))

    print(results)


def construct_next(tup):
    i, j = tup
    return (i + 1, j + 1)


if __name__ == "__main__":
    print(find_morphemes("steigen", "stiegen"))
    print(find_morphemes("steigt", "stiegst"))
    print(find_morphemes("steigst", "steigt"))
    print(find_morphemes("geben", "gegeben"))
    print(find_morphemes("machen", "gemacht"))
