Error in loadNamespace(x) : there is no package called ‘outlieR’

Traceback:
10: stop(cond)
9: doWithOneRestart(return(expr), restart)
8: withOneRestart(expr, restarts[[1L]])
7: withRestarts(stop(cond), retry_loadNamespace = function() NULL)
6: loadNamespace(x)
5: data %>% outlieR::impute(flag = NULL, fill = "mean", level = 0.1, 
       nmax = NULL, side = NULL, crit = "lof", k = 5, metric = "euclidean", 
       q = 3) at preprocess_data.r#8
4: eval(ei, envir)
3: eval(ei, envir)
2: withVisible(eval(ei, envir))
1: source("scripts/preprocess_data.r")

Warning messages:
1: In text.default(xy$x, xy$y + 0.5 * cxy[2L], rows[left.child],  ... :
  "use.n." is not a graphical parameter
2: In text.default(xy$x[leaves], xy$y[leaves] - 0.5 * cxy[2L],  ... :
  "use.n." is not a graphical parameter
3: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
4: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
5: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
6: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
7: In text.default(xy$x, xy$y + 0.5 * cxy[2L], rows[left.child],  ... :
  "use.n." is not a graphical parameter
8: In text.default(xy$x[leaves], xy$y[leaves] - 0.5 * cxy[2L],  ... :
  "use.n." is not a graphical parameter
9: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
10: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
11: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
12: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
13: In text.default(xy$x, xy$y + 0.5 * cxy[2L], rows[left.child],  ... :
  "use.n." is not a graphical parameter
14: In text.default(xy$x[leaves], xy$y[leaves] - 0.5 * cxy[2L],  ... :
  "use.n." is not a graphical parameter
15: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
16: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
17: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
18: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
19: In text.default(xy$x, xy$y + 0.5 * cxy[2L], rows[left.child],  ... :
  "use.n." is not a graphical parameter
20: In text.default(xy$x[leaves], xy$y[leaves] - 0.5 * cxy[2L],  ... :
  "use.n." is not a graphical parameter
21: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
22: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
23: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
24: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
25: In text.default(xy$x, xy$y + 0.5 * cxy[2L], rows[left.child],  ... :
  "use.n." is not a graphical parameter
26: In text.default(xy$x[leaves], xy$y[leaves] - 0.5 * cxy[2L],  ... :
  "use.n." is not a graphical parameter
27: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
28: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
29: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
30: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
31: In text.default(xy$x, xy$y + 0.5 * cxy[2L], rows[left.child],  ... :
  "use.n." is not a graphical parameter
32: In text.default(xy$x[leaves], xy$y[leaves] - 0.5 * cxy[2L],  ... :
  "use.n." is not a graphical parameter
33: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
34: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
35: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
36: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
37: In text.default(xy$x, xy$y + 0.5 * cxy[2L], rows[left.child],  ... :
  "use.n." is not a graphical parameter
38: In text.default(xy$x[leaves], xy$y[leaves] - 0.5 * cxy[2L],  ... :
  "use.n." is not a graphical parameter
39: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
40: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
41: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
42: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
43: In text.default(xy$x, xy$y + 0.5 * cxy[2L], rows[left.child],  ... :
  "use.n." is not a graphical parameter
44: In text.default(xy$x[leaves], xy$y[leaves] - 0.5 * cxy[2L],  ... :
  "use.n." is not a graphical parameter
45: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
46: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
47: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
48: In doTryCatch(return(expr), name, parentenv, handler) :
  "use.n." is not a graphical parameter
49: In text.default(xy$x, xy$y + 0.5 * cxy[2L], rows[left.child],  ... :
  "use.n." is not a graphical parameter
50: In text.default(xy$x[leaves], xy$y[leaves] - 0.5 * cxy[2L],  ... :
  "use.n." is not a graphical parameter

