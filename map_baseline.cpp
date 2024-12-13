#include <string.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <map> // balance tree inefficient with big data, time : 1.49
#include <unordered_map> // time : 0.90

/*===========perf analyse========================
Samples: 2K of event 'cpu-clock:uhppp', Event count (approx.): 645750000
  Children      Self  Command  Shared Object         Symbol
+   49.75%    49.75%  map      map                   [.] service
+   10.72%    10.72%  map      libc.so.6             [.] __memcmp_sse2
+    4.07%     0.00%  map      [unknown]             [.] 0000000000000000
+    3.25%     3.25%  map      libc.so.6             [.] __memmove_ssse3
+    3.17%     3.17%  map      libc.so.6             [.] __strlen_sse2
+    3.06%     3.06%  map      libc.so.6             [.] __memchr_sse2
+    2.98%     2.98%  map      libc.so.6             [.] malloc_consolidate
+    2.83%     2.83%  map      map                   [.] std::vector<std::__cxx11::basic_string<char,
                                                        std::char_traits<char>, std::allocator<char> >,
                                                        std::allocator<std::__cxx11::
+    2.75%     2.75%  map      libc.so.6             [.] unlink_chunk.constprop.0
+    2.28%     2.28%  map      libc.so.6             [.] _int_free
+    2.28%     2.28%  map      libc.so.6             [.] malloc
+    2.17%     0.00%  map      [unknown]             [.] 0x48fb8948530015cd
+    2.17%     0.00%  map      libstdc++.so.6.0.30   [.] std::__cxx11::basic_stringbuf<char,
                                                        std::char_traits<char>,
                                                        std::allocator<char> >::~basic_stringbuf
+    2.17%     0.00%  map      [unknown]             [.] 0x00007fde992d3010
+    1.78%     1.78%  map      libc.so.6             [.] cfree@GLIBC_2.2.5
=================================================*/

std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> result;
    result.reserve(10);

    std::string::size_type from = 0;
    std::string::size_type colon = s.find(':');

    while (colon != std::string::npos)
    {
        result.emplace_back(s, from, colon - from);
        from = colon + 1;
        colon = s.find(':', from);
    }

    result.emplace_back(s, from);

    return result;
}

void replace(std::string& s, const char* from, const char* to) {
    std::size_t pos = 0;
    std::size_t from_len = strlen(from);
    std::size_t to_len = strlen(to);

    while ((pos = s.find(from, pos)) != std::string::npos)
    {
        s.replace(pos, from_len, to);
        pos += to_len;
    }
}

std::string concatenate_tokens(const std::vector<std::string>& tokens) {
    if (tokens.empty()) return "";

    std::string result;
    size_t total_size = 0;
    for (const auto& token : tokens) {
        total_size += token.size() + 1;
    }
    result.reserve(total_size - 1);
    result = tokens[0];
    for (size_t i = 1; i < tokens.size(); ++i) {
        result += ":";
        result += tokens[i];
    }

    return result;
}

// according to perf bottleneck here
std::string service(std::string& in) {
    std::istringstream iss(in);
    std::string result;
    result.reserve(in.size());

    std::uint64_t line_count = 0;
    std::string line;

    std::unordered_map<std::string, std::string> entries;

    while (std::getline(iss, line)) {
        std::vector<std::string> tokens = tokenize(line);

        if (tokens[0] == "params") {
            replace(tokens[1], "${param}", "task7");
            replace(tokens[1], "${tag}", "performance");
            replace(tokens[1], "${id}", "TUM");
            replace(tokens[1], "${line_count}", std::to_string(line_count).c_str());

            result += concatenate_tokens(tokens) + '\n';
            ++line_count;
        }
        else if (tokens[0] == "set") {
            entries[tokens[1]] = tokens[2];
        } else if (tokens[0] == "value") {
            auto it = entries.find(tokens[1]);
            if (it != entries.end())
                result += "value:" + it->second + '\n';
            else
                result += "value:\n";

            ++line_count;
        } else {
            result += concatenate_tokens(tokens) + '\n';
            ++line_count;
        }
    }

    return result;
}



/**********************************/
/* IMPORTANT!!!                   */
/* DON'T MODIFY THE MAIN FUNCTION */
/**********************************/

int main()
{
    std::ifstream file( "records.txt" );

    if ( file )
    {
        std::ostringstream oss;
        oss << file.rdbuf();
        std::string str = oss.str();

        auto start = std::chrono::steady_clock::now();
        std::string result = service(str);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;

        std::cout << diff.count() << std::endl;
        file.close();

        std::ofstream output("output");
        if(output.is_open())
        {
            output << result;
            output.close();
        }

        file.close();
    }
    return 0;
}
