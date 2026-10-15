"""Fast unit tests for Rook Next data splitting, response masking, and model shape."""
import sys, types, unittest
import torch
from train_rook_next import data, pack, Rook, V, PAD, BOS, EOS

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for i in range(30):
            u,a,u2,a2=f"u{i}",f"a{i}",f"u2{i}",f"a2{i}"
            self.rows.extend([
                {"message_id":u,"parent_id":None,"role":"prompter","text":f"Question {i}","deleted":False},
                {"message_id":a,"parent_id":u,"role":"assistant","text":f"Answer {i}","deleted":False},
                {"message_id":u2,"parent_id":a,"role":"prompter","text":f"Follow up {i}","deleted":False},
                {"message_id":a2,"parent_id":u2,"role":"assistant","text":f"Follow-up answer {i}","deleted":False},
            ])
        self.old=sys.modules.get("datasets")
        mod=types.ModuleType("datasets"); mod.load_dataset=lambda *a,**k:self.rows
        sys.modules["datasets"]=mod
    def tearDown(self):
        if self.old is None:sys.modules.pop("datasets",None)
        else:sys.modules["datasets"]=self.old
    def test_splits_are_disjoint_by_root(self):
        tr,va,te,meta=data(7)
        roots=[{e["root"] for e in s} for s in (tr,va,te)]
        self.assertFalse(roots[0]&roots[1]);self.assertFalse(roots[0]&roots[2]);self.assertFalse(roots[1]&roots[2])
        self.assertGreater(len(tr),0);self.assertGreater(len(va),0);self.assertGreater(len(te),0)
    def test_only_answer_is_scored(self):
        e={"prompt":"User: hello\nAssistant:","answer":" hi there"}
        x,y=pack(e,128)
        self.assertTrue((y==-100).any());self.assertTrue((y!=-100).any())
        self.assertEqual(x.dtype,torch.long)
    def test_answer_targets_start_at_prompt_boundary(self):
        e={"prompt":"User: hello\nAssistant:","answer":" hi there"}
        x,y=pack(e,128)
        prompt_len=1+len(e["prompt"].encode("utf-8"))
        self.assertEqual(int(y[prompt_len-1]), ord(" "))
        self.assertTrue(torch.all(y[:prompt_len-1] == -100))
        self.assertLessEqual(len(x),128)
        self.assertEqual(len(x),len(y))

    def test_long_answer_reserves_half_context_for_prompt(self):
        e={"prompt":"User: " + "question detail " * 20 + "\nAssistant:","answer":"x"*1000}
        x,y=pack(e,128)
        self.assertEqual(len(x),len(y))
        self.assertLessEqual(len(x),128)
        # A long answer must not crowd the question down to only a few bytes.
        self.assertEqual(int((y != -100).sum()), 64)
        self.assertEqual(int((y == -100).sum()), 64)
        self.assertTrue((y!=-100).any())
        prompt_and_answer = bytes(x.tolist()[1:]).decode("utf-8", errors="ignore")
        self.assertIn("Assistant:", prompt_and_answer)
        self.assertIn("question", prompt_and_answer)

    def test_random_model_forward(self):
        m=Rook(d=48,l=2,h=4,c=32)
        x=torch.randint(0,V,(2,32)); z=m(x)
        self.assertEqual(tuple(z.shape),(2,32,V));self.assertTrue(torch.isfinite(z).all())

if __name__=="__main__":unittest.main()
