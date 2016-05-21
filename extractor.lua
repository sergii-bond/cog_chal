require 'nn'
require 'cudnn'
require 'cunn'
require 'sys'
require 'hdf5'
local Image = require 'image'
require 'paths'
local Threads = require 'threads'
local t = require './transforms'

--------------------------------------------------------------------
-- Arguments
-- it is assumed all files in the directory are images
--local im_dir = "flickr_photos"
local im_dir = "modified_images"
-- number of feature vectors per image (transformations & crops used)
local nCrops = 1
-- model
local model_path = "resnet-101.t7"
--local out_file = "flickr_features-101.t7"
local out_file = "modified_features-101.t7"
-- number of threads for data processing
local nThreads = 7
-- batch size
local batchSize = 24 

local im_input_size = 224 -- for resnet
local scale_size = 224 -- for resnet
--------------------------------------------------------------------

local im_paths_tmp = paths.dir(im_dir)

local im_paths = {}

for i, x in ipairs(im_paths_tmp) do 
 if x ~= '.' and x ~= '..' then
  table.insert(im_paths, x)
 end
end

--print(#im_paths)

print(('Found %d images in %s'):format(#im_paths, im_dir))

-- Load the model
local model = torch.load(model_path)
print(model)

-- Remove the fully connected layer
local linear_layer = model:get(#model.modules)
local linear_layer_input_size = linear_layer.weight:size(2)
assert(torch.type(linear_layer) == 'nn.Linear')
model:remove(#model.modules)

model:evaluate()
print(linear_layer_input_size)
--assert(linear_layer_input_size == 512, 'wrong output size')

features = torch.FloatTensor(#im_paths, nCrops, linear_layer_input_size):zero()
print('Size of features Tensor:')
--print(('(%d, %d, %d)'):format(#im_paths, nCrops, linear_layer_input_size))
print(features:size())

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

-- the first transformation of this type
local t1 = t.Compose{
  t.Scale(scale_size),
  t.ColorNormalize(meanstd),
  t.CenterCrop(im_input_size),
}

-- deal with threads
local function init()
  local Image = require 'image'
end

local function main(idx)
  return 1
end

local threads = Threads(nThreads, init, main)

function run(batchSize, im_paths, nCrops)

    local batches = {}
    local batch = {}

    for i, im_path in ipairs(im_paths) do

      table.insert(batch, im_path)

      if i % batchSize == 0 then
        table.insert(batches, batch)
        batch = {}
      end
    end

    if #batch > 0 then
        table.insert(batches, batch)
    end




   local idx, sample = 1, nil

   local function enqueue()
      while idx <= #batches and threads:acceptsjob() do

          local im_paths = batches[idx]

          threads:addjob(
              function()
                local Image = require 'image'
                local batch, imageSize

                for i, im_path in ipairs(im_paths) do
                
                  -- load the image as a RGB float tensor with values 0..1
                  path = paths.concat(im_dir, im_path)
                  assert(paths.filep(path), "File doesn't exist: " .. path)
                  local img = Image.load(path, 3, 'float')

                  img_p = t1(img)

                  -- View as mini-batch of size 1
                  --img_v = img_p:view(1, table.unpack(img_p:size():totable()))
                  if not batch then
                      imageSize = img_p:size():totable()
                      if nCrops > 1 then table.remove(imageSize, 1) end
                      batch = torch.FloatTensor(#im_paths, nCrops, table.unpack(imageSize))
                  end
                  --print(img_p:size())
                  --print(batch:size())
                  batch[i]:copy(img_p)

                end

                collectgarbage()
                return {
                    input = batch:view(#im_paths * nCrops, table.unpack(imageSize)),
                    paths = im_paths
                }
              end,
              function(_sample_)
                sample = _sample_
              end
          )
         idx = idx + 1
      end
   end

   local k = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      k = k + 1
      return k, #batches, sample
   end

   return loop
end

local pfreq = 10

local new_ordered_paths = {}
local k = 1

--the order in original im_paths is not preserved
for i, N, batch in run(batchSize, im_paths, nCrops) do
  for _, x in ipairs(batch.paths) do
    table.insert(new_ordered_paths, x) 
  end

  local output = model:forward(batch.input:cuda())
  local x = output:float()
  local y = x:resize(x:size(1) / nCrops, nCrops, x:size(2))

  for j = 1, y:size(1) do
    features[{{k}, {}, {}}]:copy(y[j])
    k = k + 1
  end

  if i % pfreq == 0 then
     print('Processed ' .. i)
  end
end


df = {im_path2id = new_ordered_paths, features = features} 

print('Saving normalized features to disk, ' .. out_file)
  torch.save(out_file, df)
print('Done.')

