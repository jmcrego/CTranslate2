# coding: utf-8

import ctranslate2
import numpy as np
import os
import sys
import logging
from time import time
import concurrent.futures

def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.info('Created Logger level={}'.format(loglevel))
    else:
        logging.basicConfig(filename=logfile, format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.info('Created Logger level={} file={}'.format(loglevel, logfile))

def read_file(f, is_prefix=False, source_and_prefix=False):
    if source_and_prefix:
        x = [l.rstrip('\n') for l in open(f)]
        v = []
        w = []
        for i in range(len(x)):
            src_pre = x[i].split('\t')
            if len(src_pre) != 2:
                logging.error('line {} without [tab] separating source/prefix >{}<'.format(len(v)+1,x[i]))
                sys.exit()
            v.append(src_pre[0].split())
            w.append(src_pre[1].split())
            if len(w[-1]) == 0: #empty prefix
                w[-1] = None

        return v,w
    
    v = [l.rstrip('\n').split() for l in open(f)]
    if is_prefix:
        for i in range(len(v)):
            if len(v[i]) == 0: #empty prefix
                v[i] = None
    return v
    

################################################################
### Args #######################################################
################################################################

class Args():

    def __init__(self, argv):    
        self.fsource = None
        self.fprefix = None
        self.fsrcpref = None
        self.fmodel = None
        self.tok_prefix = '‖'
        self.device = 'cpu'
        self.inter_threads = 16
        self.intra_threads = 1
        self.max_batch_size = 60
        self.beam_size = 5
        self.length_penalty = 0.6
        self.min_decoding_length = 0

        log_file = None
        log_level = 'debug'
        prog = argv.pop(0)
        usage = '''usage: {} -m PATH -s FILE [-p FILE] [-sp FILE]
    -m  PATH : model path
    -s  FILE : source file
    -p  FILE : prefix file (prefix lines must be ended by tok_prefix)
    -sp FILE : source and prefix file (lines include source and prefix separateds by tab)

    -tok_prefix          STR : token used to mark end of prefix (default ‖)
    -inter_threads       INT : inter threads (default 16)
    -intra_threads       INT : intra threads (default 1)
    -max_batch_size      INT : max batch size (default 60)
    -beam_size           INT : beam size (default 5)
    -length_penalty      INT : length penalty (default 0.6)
    -min_decoding_length INT : min decoding length (default 0)
    -device              STR : device to translate [cpu, cuda, auto] (default cpu)
    -log_level           STR : logging level [debug, info, warning, critical, error] (default debug)
    -log_file           FILE : logging file (default stderr)
    -h                       : this help

ATTENTION: Convert (export) model previous to translate. Use:
 onmt-main --config config.yml --auto_config export --export_dir <model path> --export_format ctranslate2

'''.format(prog)

        while len(argv):
            tok = argv.pop(0)
            if tok=="-h":
                sys.stderr.write("{}".format(usage))
                sys.exit()
            elif tok=="-s" and len(argv):
                self.fsource = argv.pop(0)
            elif tok=="-p" and len(argv):
                self.fprefix = argv.pop(0)
            elif tok=="-sp" and len(argv):
                self.fsrcpref = argv.pop(0)
            elif tok=="-m" and len(argv):
                self.fmodel = argv.pop(0)
            elif tok=="-tok_prefix" and len(argv):
                self.tok_prefix = argv.pop(0)
            elif tok=="-inter_threads" and len(argv):
                self.inter_threads = int(argv.pop(0))
            elif tok=="-intra_threads" and len(argv):
                self.intra_threads = int(argv.pop(0))
            elif tok=="-max_batch_size" and len(argv):
                self.max_batch_size = int(argv.pop(0))
            elif tok=="-beam_size" and len(argv):
                self.beam_size = int(argv.pop(0))
            elif tok=="-length_penalty" and len(argv):
                self.length_penalty = float(argv.pop(0))
            elif tok=="-min_decoding_length" and len(argv):
                self.min_decoding_length = int(argv.pop(0))
            elif tok=="-device" and len(argv):
                self.device = argv.pop(0)
            elif tok=="-log_file" and len(argv):
                log_file = argv.pop(0)
            elif tok=="-log_level" and len(argv):
                log_level = argv.pop(0)
            else:
                sys.stderr.write('error: unparsed {} option\n'.format(tok))
                sys.stderr.write("{}".format(usage))
                sys.exit()

        create_logger(log_file,log_level)

        if self.fmodel is None:
            logging.error('error: missing -m option')
            sys.exit()

        if self.fsource is None and self.fsrcpref is None:
            logging.error('error: missing one of -s or -sp options')
            sys.exit()

        logging.debug('tok_prefix={}'.format(self.tok_prefix))
        logging.debug('inter_threads={}'.format(self.inter_threads))
        logging.debug('intra_threads={}'.format(self.intra_threads))
        logging.debug('max_batch_size={}'.format(self.max_batch_size))
        logging.debug('beam_size={}'.format(self.beam_size))
        logging.debug('length_penalty={}'.format(self.length_penalty))
        logging.debug('min_decoding_length={}'.format(self.min_decoding_length))
        logging.debug('device={}'.format(self.device))
        logging.debug('model={}'.format(self.fmodel))
        logging.debug('source={}'.format(self.fsource))
        logging.debug('prefix={}'.format(self.fprefix))
        logging.debug('sourceprefix={}'.format(self.fsrcpref))

############################################################################
### MAIN ###################################################################
############################################################################

if __name__ == "__main__":
    args = Args(sys.argv) #creates logger

    source = None
    prefix = None
    if args.fsrcpref is not None:
        source, prefix = read_file(args.fsrcpref, source_and_prefix=True)
        logging.info('read source/prefix with {} lines'.format(len(source)))

    if args.fsource is not None:
        source = read_file(args.fsource)
        logging.info('read source with {} lines'.format(len(source)))

    if args.fprefix is not None:
        prefix = read_file(args.fprefix, is_prefix=True)
        if len(source) != len(prefix):
            logging.error('len(source) != len(prefix)')
            sys.exit()
        logging.info('read prefix with {} lines'.format(len(prefix)))

    if prefix is None:
        prefix = [None for _ in range(len(source))]
        logging.info('no prefix used')

    translator = ctranslate2.Translator(model_path=args.fmodel, device=args.device, inter_threads=args.inter_threads, intra_threads=args.intra_threads)
    logging.info('Built translator')

    def translate(translator, src, pref, args):
        return translator.translate_batch(source=src, target_prefix=pref, max_batch_size=args.max_batch_size, beam_size=args.beam_size, length_penalty=args.length_penalty, min_decoding_length=args.min_decoding_length)
    
    batch_size = len(source) // args.inter_threads
    logging.info('Start batch_size={}'.format(batch_size))
    tic = time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.inter_threads) as executor:
        future_to_trans = [executor.submit(translate, translator, source[i: i + batch_size], prefix[i: i + batch_size], args) for i in range(0, len(source), batch_size)]
        logging.info('({} batchs submitted)'.format(len(future_to_trans)))
        b = 0
        n = 0
        for future in future_to_trans:
            batch = future.result()
            b += 1
            logging.info('(translated batch {} with {} examples)'.format(b,len(batch)))
            for line in batch:
                print('MySRC[{}]: {}'.format(n,' '.join(source[n])))
                if len(prefix[n]):
                    print('MyPRF[{}]: {}'.format(n,' '.join(prefix[n])))
                print('MyOUT[{}]: {}'.format(n,' '.join(line[0]['tokens'])))
                if len(line):
                    hyp = " ".join(line[0]["tokens"])
                    #hyp = hyp.split(' '+args.tok_prefix+' ')[-1]
                    hyp = hyp.split(args.tok_prefix+' ')[-1]
                else:
                    hyp = ""
                print(hyp)
                n += 1

    toc = time()
    logging.info("End ({:.2f} seconds [{:.2f} sents/sec])".format(toc-tic, 1.0*len(source)/(toc-tic)))


