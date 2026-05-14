#!/usr/bin/env python3
"""
SilkLoom Core v4.1 优化功能测试脚本
验证新增功能的基本可用性
"""

import sys
import os

# 将 silkloom_core 加入路径
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """测试 1: 验证新增模块能否正常导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)
    
    try:
        from silkloom_core import (
            PromptMapper, 
            ResultSet, 
            TaskResult,
            CacheManager,
            SilkLoomError,
        )
        print("✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_template_extraction():
    """测试 2: 验证模板变量自动提取"""
    print("\n" + "=" * 60)
    print("测试 2: 模板变量提取")
    print("=" * 60)
    
    try:
        from silkloom_core import PromptMapper
        
        # 测试 2a: 标准 Jinja2 变量提取
        mapper = PromptMapper(
            model="gpt-4",
            user_prompt="分析文章：{{ title }}\n内容：{{ content }}\n作者：{{ author }}"
        )
        
        required_vars = mapper.required_variables
        expected = {'title', 'content', 'author'}
        
        if required_vars == expected:
            print(f"✓ 变量提取成功: {required_vars}")
        else:
            print(f"✗ 变量提取失败: 期望 {expected}, 得到 {required_vars}")
            return False
        
        # 测试 2b: 简单模式变量提取
        mapper_simple = PromptMapper(
            model="gpt-4",
            user_prompt="分析文章：{title}\n内容：{content}",
            simple_format_mode=True
        )
        
        required_vars_simple = mapper_simple.required_variables
        expected_simple = {'title', 'content'}
        
        if required_vars_simple == expected_simple:
            print(f"✓ 简单模式变量提取成功: {required_vars_simple}")
        else:
            print(f"✗ 简单模式失败: 期望 {expected_simple}, 得到 {required_vars_simple}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_result():
    """测试 3: 验证 TaskResult 和链式操作"""
    print("\n" + "=" * 60)
    print("测试 3: ResultSet 链式操作")
    print("=" * 60)
    
    try:
        from silkloom_core import TaskResult, ResultSet
        
        # 创建测试数据
        results_list = [
            TaskResult(is_success=True, data="Hello", usage={"total_tokens": 100}),
            TaskResult(is_success=False, error="Failed"),
            TaskResult(is_success=True, data="World", usage={"total_tokens": 150}),
        ]
        
        rs = ResultSet(results_list, run_id="test_001")
        
        # 测试基本属性
        print(f"✓ 总任务数: {len(rs)}")
        print(f"✓ 成功: {rs.success_count}, 失败: {rs.failed_count}")
        
        # 测试 filter
        successful = rs.filter(lambda r: r.is_success)
        if len(successful) == 2:
            print(f"✓ filter() 成功，过滤出 {len(successful)} 个成功任务")
        else:
            print(f"✗ filter() 失败: 期望 2, 得到 {len(successful)}")
            return False
        
        # 测试 map
        tokens = rs.map(lambda r: r.usage.get('total_tokens', 0) if r.usage else 0)
        if tokens == [100, 0, 150]:
            print(f"✓ map() 成功，提取 tokens: {tokens}")
        else:
            print(f"✗ map() 失败: 期望 [100, 0, 150], 得到 {tokens}")
            return False
        
        # 测试 transform
        upper_rs = rs.transform(lambda d: d.upper() if isinstance(d, str) else d)
        if upper_rs.results[0].data == "HELLO":
            print(f"✓ transform() 成功")
        else:
            print(f"✗ transform() 失败")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_manager():
    """测试 4: 验证 CacheManager 基本功能"""
    print("\n" + "=" * 60)
    print("测试 4: CacheManager")
    print("=" * 60)
    
    try:
        from silkloom_core import CacheManager
        import tempfile
        import os
        
        # 使用临时数据库避免污染
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            cache = CacheManager(db_path=db_path)
            
            try:
                # 测试 inspect（空数据库）
                summary = cache.inspect()
                if "run_summaries" in summary:
                    print(f"✓ CacheManager.inspect() 成功")
                else:
                    print(f"✗ inspect() 返回格式错误")
                    return False
            finally:
                # 确保关闭数据库连接
                import gc
                gc.collect()
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exception_messages():
    """测试 5: 验证异常消息改进"""
    print("\n" + "=" * 60)
    print("测试 5: 异常消息")
    print("=" * 60)
    
    try:
        from silkloom_core import PromptMapper, ConfigurationError
        
        # 测试配置错误消息
        try:
            mapper = PromptMapper(
                model="gpt-4",
                user_prompt="{text}",
                max_retries=0  # ❌ 应该抛错
            )
            print("✗ 应该抛出 ConfigurationError")
            return False
        except ConfigurationError as e:
            if "max_retries" in str(e):
                print(f"✓ ConfigurationError 消息清晰: {e}")
            else:
                print(f"✗ 错误消息不够详细: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  SilkLoom Core v4.1 优化功能测试                    ║")
    print("╚" + "=" * 58 + "╝")
    
    tests = [
        ("模块导入", test_imports),
        ("模板变量提取", test_template_extraction),
        ("ResultSet 链式操作", test_task_result),
        ("CacheManager", test_cache_manager),
        ("异常消息", test_exception_messages),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} 出错: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n总体: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！优化功能可用。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
