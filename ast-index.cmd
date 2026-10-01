@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "AST_INDEX_CACHE_DIR=%~dp0.analysis\ast-index-cache"
if not exist "tools\ast-index\node_modules\.bin\ast-index.cmd" (
  echo AST Index is not installed. Run from the project root: >&2
  echo npm.cmd ci --prefix tools/ast-index --cache .analysis/npm-cache >&2
  popd
  exit /b 127
)
call "tools\ast-index\node_modules\.bin\ast-index.cmd" %*
set "ast_index_exit=%errorlevel%"
popd
exit /b %ast_index_exit%
