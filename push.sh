#!/bin/bash

git add --all
git commit -m 'package update'
git push origin main
git push origin :v0.0.1
git push origin v0.0.1
