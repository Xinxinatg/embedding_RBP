# https://www.biostars.org/p/4881/
import sys
from Bio import SeqIO

fasta_file = sys.argv[1]  # Input fasta file
remove_file = sys.argv[2] # Input wanted file, one gene name per line
result_file = sys.argv[3] # Output fasta file

remove = []
# with open(remove_file) as f:
#     for line in f:
#         line = line.strip()
#         if line != "" and '>' not in line:
#             remove.add(line)
# print(len(remove))
rm_sequences = SeqIO.parse(open(remove_file),'fasta')
for rm_seq in rm_sequences:
    remove.append(str(rm_seq.seq))
print(len(remove))
fasta_sequences = SeqIO.parse(open(fasta_file),'fasta')

with open(result_file, "w") as f:
    for seq in fasta_sequences:
        nuc = str(seq.seq)
        if nuc not in remove and len(nuc) > 0:
            SeqIO.write([seq], f, "fasta")
