
require 'sys'
require 'paths'
local csv = require 'csvigo'

--------------------------------------------------------------------------------------------
local modified_file = "modified_features-101.t7"
local flickr_file = "flickr_features-101.t7"
local match_dic = paths.concat('', 'match_dic.csv')
--------------------------------------------------------------------------------------------

-- A - matrix of size N1xD = N1 vectors of dim D
-- B - matrix of size N2xD
-- returns N1xN2 matrix, where (i, j) entry is a norm of (vec i - vec j)
function norm_of_diff(A, B)
  assert(A:size(2) == B:size(2), 'vector dims must match')
  local NA = A:size(1)
  local NB = B:size(1)
  local A2 = torch.sum(torch.pow(A, 2), 2)
  local B2 = torch.sum(torch.pow(B, 2), 2)
  local A2_1 = torch.cat(A2, torch.ones(NA, 1), 2)
  local B2_1 = torch.cat(torch.ones(1, NB), B2:transpose(1, 2), 1)
  local AB2 = torch.mm(A2_1, B2_1)
  local AB = torch.mul(torch.mm(A, -B:transpose(1, 2)), 2)
  local C = AB2 + AB
  -- abs to avoid tiny negative diagonal elements

  return torch.sqrt(torch.abs(C))
end

local timer = torch.Timer()

assert(paths.filep(modified_file), 'File not found: ' .. modified_file)
print('Loading features from file: ' .. modified_file)
local df_mod = torch.load(modified_file)

assert(paths.filep(flickr_file), 'File not found: ' .. flickr_file)
print('Loading features from file: ' .. flickr_file)
local df_ori = torch.load(flickr_file)

--print(df_mod.features:size())
--print(df_mod.features:view(df_mod.features:size(1), df_mod.features:size(3)):size())
--print(df_mod.im_path2id[1])
local mod_f = df_mod.features:view(df_mod.features:size(1), df_mod.features:size(3))
local ori_f = df_ori.features:view(df_ori.features:size(1), df_ori.features:size(3))
local nd = norm_of_diff(mod_f:double(), ori_f:double())  
local y, i = torch.min(nd, 2)

print(df_mod.im_path2id[3])
print(df_ori.im_path2id[i[3][1]])


local new_csv_dic = {mod = {}, ori = {}}

for id = 1, #df_mod.im_path2id, 1 do
  table.insert(new_csv_dic['mod'], df_mod.im_path2id[id])
  table.insert(new_csv_dic['ori'], df_ori.im_path2id[i[id][1]])
end

csv.save(match_dic, new_csv_dic)
os.exit()


print('Run Time : ' .. timer:time().real)
