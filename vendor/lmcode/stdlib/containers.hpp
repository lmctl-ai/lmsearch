#pragma once
// Finish the default aggregator only after this group's types are complete.
// Result conversion needs IntMap/FloatMap even on a standalone group include.
#ifndef DSLVM_STD_BUILDING
#define DSLVM_STD_DEFER
#include "dslvm.hpp"
#undef DSLVM_STD_DEFER
#define DSLVM_STD_BUILDING
#define DSLVM_CONTAINERS_FINISH_STD
#endif
#include "collections.hpp"
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dslvm::stdlib {
template <class T> struct TypedMap {
  std::vector<std::pair<std::string, T>> items;
  std::unordered_map<std::string, size_t> index;
  void set(const std::string &k, T v) {
    auto i = index.find(k);
    if (i != index.end()) {
      items[i->second].second = std::move(v);
      return;
    }
    index.emplace(k, items.size());
    items.emplace_back(k, std::move(v));
  }
  void clear() {
    items.clear();
    index.clear();
  }
};
using IntMap = TypedMap<int64_t>;
using FloatMap = TypedMap<double>;
template <class T> struct Stack { std::vector<T> items; };
template <class T> struct Queue { std::deque<T> items; };
template <class T> struct PriorityQueue {
  struct Item {
    T value;
    double priority;
    int64_t sequence;
  };
  std::vector<Item> items;
  int64_t next_sequence = 0;
};
using IntStack = Stack<int64_t>;
using FloatStack = Stack<double>;
using StrStack = Stack<std::string>;
using IntQueue = Queue<int64_t>;
using FloatQueue = Queue<double>;
using StrQueue = Queue<std::string>;
using IntPQ = PriorityQueue<int64_t>;
using FloatPQ = PriorityQueue<double>;
using StrPQ = PriorityQueue<std::string>;
namespace detail {
template <class T>
inline void map(Registry &r, const std::string &t, const std::string &et) {
  using M = TypedMap<T>;
  using X = CollectionValue<T>;
  r.method(t, "len", {}, "int", [](void *p, Args &) {
    return Value::integer(
        static_cast<int64_t>(static_cast<M *>(p)->items.size()));
  });
  r.method(t, "has", {"string"}, "bool", [](void *p, Args &a) {
    return Value::boolean(static_cast<M *>(p)->index.count(a[0].s));
  });
  r.method(t, "get", {"string"}, et, [t](void *p, Args &a) {
    auto &m = *static_cast<M *>(p);
    auto i = m.index.find(a[0].s);
    if (i == m.index.end())
      throw Error(t + ".get: missing key");
    return X::put(m.items[i->second].second);
  });
  r.method(t, "get_or", {"string", et}, et, [](void *p, Args &a) {
    auto &m = *static_cast<M *>(p);
    auto i = m.index.find(a[0].s);
    return i == m.index.end() ? a[1] : X::put(m.items[i->second].second);
  });
  r.method(t, "set", {"string", et}, t, [t](void *p, Args &a) {
    auto &m = *static_cast<M *>(p);
    m.set(a[0].s, X::get(a[1]));
    return Value::object(t, p);
  });
  r.method(t, "key_at", {"int"}, "string", [t](void *p, Args &a) {
    auto &m = *static_cast<M *>(p);
    auto i = collection_index(a[0].i, m.items.size(), t + ".key_at");
    return Value::string(m.items[i].first);
  });
  r.method(t, "remove", {"string"}, "bool", [](void *p, Args &a) {
    auto &m = *static_cast<M *>(p);
    auto i = m.index.find(a[0].s);
    if (i == m.index.end())
      return Value::boolean(false);
    size_t n = i->second;
    m.index.erase(i);
    m.items.erase(m.items.begin() + n);
    for (size_t j = n; j < m.items.size(); ++j)
      m.index[m.items[j].first] = j;
    return Value::boolean(true);
  });
  r.method(t, "clear", {}, t, [t](void *p, Args &) {
    static_cast<M *>(p)->clear();
    return Value::object(t, p);
  });
}
template <class T>
inline void map_snapshots(Registry &r, const std::string &t, Collections &a) {
  using M = TypedMap<T>;
  r.method(t, "keys", {}, "StrList", [&a](void *p, Args &) {
    auto &m = *static_cast<M *>(p);
    std::vector<std::string> values; values.reserve(m.items.size());
    for (const auto &item : m.items) values.push_back(item.first);
    return Value::object("StrList", a.make_str_list(std::move(values)));
  });
  r.method(t, "values", {}, std::is_same_v<T, std::string> ? "StrList" : (std::is_same_v<T, int64_t> ? "IntList" : "FloatList"), [&a](void *p, Args &) {
    auto &m = *static_cast<M *>(p);
    if constexpr (std::is_same_v<T, std::string>) {
      std::vector<std::string> values; values.reserve(m.items.size());
      for (const auto &item : m.items) values.push_back(item.second);
      return Value::object("StrList", a.make_str_list(std::move(values)));
    } else if constexpr (std::is_same_v<T, int64_t>) {
      std::vector<int64_t> values; values.reserve(m.items.size());
      for (const auto &item : m.items) values.push_back(item.second);
      return Value::object("IntList", a.make_int_list(std::move(values)));
    } else {
      std::vector<double> values; values.reserve(m.items.size());
      for (const auto &item : m.items) values.push_back(item.second);
      return Value::object("FloatList", a.make_float_list(std::move(values)));
    }
  });
}
template <class T>
inline void stack(Registry &r, const std::string &t, const std::string &et) {
  using S = Stack<T>;
  using X = CollectionValue<T>;
  r.method(t, "push", {et}, t, [t](void *p, Args &a) {
    static_cast<S *>(p)->items.push_back(X::get(a[0]));
    return Value::object(t, p);
  });
  r.method(t, "pop", {}, et, [t](void *p, Args &) {
    auto &s = *static_cast<S *>(p);
    if (s.items.empty())
      throw Error(t + ".pop: empty");
    T v = std::move(s.items.back());
    s.items.pop_back();
    return X::put(v);
  });
  r.method(t, "peek", {}, et, [t](void *p, Args &) {
    auto &s = *static_cast<S *>(p);
    if (s.items.empty())
      throw Error(t + ".peek: empty");
    return X::put(s.items.back());
  });
  r.method(t, "len", {}, "int", [](void *p, Args &) {
    return Value::integer((int64_t) static_cast<S *>(p)->items.size());
  });
  r.method(t, "empty", {}, "bool", [](void *p, Args &) {
    return Value::boolean(static_cast<S *>(p)->items.empty());
  });
  r.method(t, "clear", {}, t, [t](void *p, Args &) {
    static_cast<S *>(p)->items.clear();
    return Value::object(t, p);
  });
  r.method(t, "at", {"int"}, et, [t](void *p, Args &a) {
    auto &s = *static_cast<S *>(p);
    auto i = collection_index(a[0].i, s.items.size(), t + ".at");
    return X::put(s.items[s.items.size() - 1 - i]);
  });
}
template <class T>
inline void queue(Registry &r, const std::string &t, const std::string &et) {
  using Q = Queue<T>;
  using X = CollectionValue<T>;
  r.method(t, "push", {et}, t, [t](void *p, Args &a) {
    static_cast<Q *>(p)->items.push_back(X::get(a[0]));
    return Value::object(t, p);
  });
  r.method(t, "pop", {}, et, [t](void *p, Args &) {
    auto &q = *static_cast<Q *>(p);
    if (q.items.empty())
      throw Error(t + ".pop: empty");
    T v = std::move(q.items.front());
    q.items.pop_front();
    return X::put(v);
  });
  r.method(t, "peek", {}, et, [t](void *p, Args &) {
    auto &q = *static_cast<Q *>(p);
    if (q.items.empty())
      throw Error(t + ".peek: empty");
    return X::put(q.items.front());
  });
  r.method(t, "len", {}, "int", [](void *p, Args &) {
    return Value::integer((int64_t) static_cast<Q *>(p)->items.size());
  });
  r.method(t, "empty", {}, "bool", [](void *p, Args &) {
    return Value::boolean(static_cast<Q *>(p)->items.empty());
  });
  r.method(t, "clear", {}, t, [t](void *p, Args &) {
    static_cast<Q *>(p)->items.clear();
    return Value::object(t, p);
  });
  r.method(t, "at", {"int"}, et, [t](void *p, Args &a) {
    auto &q = *static_cast<Q *>(p);
    return X::put(q.items[collection_index(a[0].i, q.items.size(), t + ".at")]);
  });
}
template <class T>
inline void pq(Registry &r, const std::string &t, const std::string &et) {
  using P = PriorityQueue<T>;
  using X = CollectionValue<T>;
  auto worse = [](const typename P::Item &a, const typename P::Item &b) {
    return a.priority > b.priority ||
           (a.priority == b.priority && a.sequence > b.sequence);
  };
  r.method(t, "push", {et, "float"}, t, [t, worse](void *p, Args &a) {
    auto &q = *static_cast<P *>(p);
    if (!std::isfinite(a[1].f))
      throw Error(t + ".push: priority must be finite");
    if (q.next_sequence == std::numeric_limits<int64_t>::max())
      throw Error(t + ".push: insertion sequence exhausted");
    const int64_t sequence = q.next_sequence;
    q.items.push_back({X::get(a[0]), a[1].f, sequence});
    std::push_heap(q.items.begin(), q.items.end(), worse);
    q.next_sequence = sequence + 1;
    return Value::object(t, p);
  });
  r.method(t, "peek", {}, et, [t](void *p, Args &) {
    auto &q = *static_cast<P *>(p);
    if (q.items.empty())
      throw Error(t + ".peek: empty");
    return X::put(q.items.front().value);
  });
  r.method(t, "pop", {}, et, [t, worse](void *p, Args &) {
    auto &q = *static_cast<P *>(p);
    if (q.items.empty())
      throw Error(t + ".pop: empty");
    std::pop_heap(q.items.begin(), q.items.end(), worse);
    T v = std::move(q.items.back().value);
    q.items.pop_back();
    return X::put(v);
  });
  r.method(t, "len", {}, "int", [](void *p, Args &) {
    return Value::integer((int64_t) static_cast<P *>(p)->items.size());
  });
  r.method(t, "empty", {}, "bool", [](void *p, Args &) {
    return Value::boolean(static_cast<P *>(p)->items.empty());
  });
  r.method(t, "clear", {}, t, [t](void *p, Args &) {
    auto &q = *static_cast<P *>(p);
    q.items.clear();
    q.next_sequence = 0;
    return Value::object(t, p);
  });
}
} // namespace detail
inline Registry &register_containers(Registry &r, Collections &a) {
  detail::map<int64_t>(r, "IntMap", "int");
  detail::map<double>(r, "FloatMap", "float");
  detail::map_snapshots<int64_t>(r, "IntMap", a);
  detail::map_snapshots<double>(r, "FloatMap", a);
  detail::stack<int64_t>(r, "IntStack", "int");
  detail::stack<double>(r, "FloatStack", "float");
  detail::stack<std::string>(r, "StrStack", "string");
  detail::queue<int64_t>(r, "IntQueue", "int");
  detail::queue<double>(r, "FloatQueue", "float");
  detail::queue<std::string>(r, "StrQueue", "string");
  detail::pq<int64_t>(r, "IntPQ", "int");
  detail::pq<double>(r, "FloatPQ", "float");
  detail::pq<std::string>(r, "StrPQ", "string");
  r.function("int_map", {}, "IntMap", [&a](Args &) {
    return Value::object("IntMap", a.make_object<IntMap>("IntMap"));
  });
  r.function("float_map", {}, "FloatMap", [&a](Args &) {
    return Value::object("FloatMap", a.make_object<FloatMap>("FloatMap"));
  });
  r.function("int_stack", {}, "IntStack", [&a](Args &) {
    return Value::object("IntStack", a.make_object<IntStack>("IntStack"));
  });
  r.function("float_stack", {}, "FloatStack", [&a](Args &) {
    return Value::object("FloatStack", a.make_object<FloatStack>("FloatStack"));
  });
  r.function("str_stack", {}, "StrStack", [&a](Args &) {
    return Value::object("StrStack", a.make_object<StrStack>("StrStack"));
  });
  r.function("int_queue", {}, "IntQueue", [&a](Args &) {
    return Value::object("IntQueue", a.make_object<IntQueue>("IntQueue"));
  });
  r.function("float_queue", {}, "FloatQueue", [&a](Args &) {
    return Value::object("FloatQueue", a.make_object<FloatQueue>("FloatQueue"));
  });
  r.function("str_queue", {}, "StrQueue", [&a](Args &) {
    return Value::object("StrQueue", a.make_object<StrQueue>("StrQueue"));
  });
  r.function("int_pq", {}, "IntPQ", [&a](Args &) {
    return Value::object("IntPQ", a.make_object<IntPQ>("IntPQ"));
  });
  r.function("float_pq", {}, "FloatPQ", [&a](Args &) {
    return Value::object("FloatPQ", a.make_object<FloatPQ>("FloatPQ"));
  });
  r.function("str_pq", {}, "StrPQ", [&a](Args &) {
    return Value::object("StrPQ", a.make_object<StrPQ>("StrPQ"));
  });
  return r;
}
} // namespace dslvm::stdlib

#ifdef DSLVM_CONTAINERS_FINISH_STD
#undef DSLVM_CONTAINERS_FINISH_STD
#undef DSLVM_STD_BUILDING
#include "std.hpp"
#endif
